import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from dotenv import load_dotenv
load_dotenv()


import requests
import shutil
import subprocess
import glob
import soundfile as sf
import time
import re 
from contextlib import asynccontextmanager 
from fastapi import FastAPI
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse 
from pydantic import BaseModel
from supabase import create_client
from urllib.parse import urlparse
from datetime import datetime
from zoneinfo import ZoneInfo

# DL Libraries
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import librosa
import scipy.io
import matplotlib.pyplot as plt


# Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Testing the connection
# print("🔍 SUPABASE_URL =", os.getenv("SUPABASE_URL"))
# print("🔐 SUPABASE_SERVICE_ROLE_KEY =", os.getenv("SUPABASE_SERVICE_ROLE_KEY")[:5] + '...')  # Hide most of the key
PN_CODE_LENGTH = 1000
# --- THE ATTACK LIST  ---
ATTACK_LIST = [
    ([0, 0], 'Clean'), # Class 0
    ([1, 3000], 'LPF3k'), ([1, 6000], 'LPF6k'), ([1, 9000], 'LPF9k'),
    ([2, 1], 'BPF100-3k'), ([2, 2], 'BPF100-6k'), ([2, 3], 'BPF100-9k'),
    ([2, 4], 'BPF50-6k'), ([2, 5], 'BPF25-6k'),
    ([3, 8], 'Requantization8bit'),
    ([5, 10], 'AdditiveWhite10dB'), ([5, 20], 'AdditiveWhite20dB'), ([5, 30], 'AdditiveWhite30dB'),
    ([6, 1], 'Resampling11k'), ([6, 2], 'Resampling16k'), ([6, 3], 'Resampling22k'), ([6, 4], 'Resampling24k'),
    ([7, 1], 'TimeScale0.99'), ([7, 2], 'TimeScale0.98'), ([7, 3], 'TimeScale0.97'), ([7, 4], 'TimeScale0.96'),
    ([8, 1], 'LinearSpeed0.99'), ([8, 2], 'LinearSpeed0.95'), ([8, 3], 'LinearSpeed0.9'),
    ([9, 1], 'PitchShift0.99'), ([9, 2], 'PitchShift0.98'), ([9, 3], 'PitchShift0.97'), ([9, 4], 'PitchShift0.96'),
    ([10, 1], 'Equalizer'),
    ([11, 1], 'Echo'),
    ([13, 32], 'MP3_32k'), ([13, 64], 'MP3_64k'), ([13, 96], 'MP3_96k'),
    ([13, 128], 'MP3_128k'), ([13, 192], 'MP3_192k')
]
NUM_ATTACK_CLASSES = len(ATTACK_LIST)
id_to_attack_name = {i: attack[1] for i, attack in enumerate(ATTACK_LIST)}

DL_MODEL = None
# MODEL_PATH = "models/MultiTaskExtractor_v111_BEST.h5" # Use server_21Layer.py for this model
# MODEL_PATH = "models/MultiTaskExtractor_v15_BEST.h5" # Use server_21Layer.py for this model
# MODEL_PATH = "models/MultiTaskExtractor_v11_BEST.h5"
MODEL_PATH = "models/MultiTaskExtractor_v15_Copy.keras"
# MODEL_PATH = "models/MultiTaskExtractor_v11_BEST_Final.keras"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global DL_MODEL
    print("🚀 Server starting up...")

    try:
        if os.path.exists(MODEL_PATH):
            print(f"🔍 Loading full model from: {os.path.abspath(MODEL_PATH)}")
            
            # ✅ Load full model directly (architecture + weights + config)
            DL_MODEL = keras.models.load_model(MODEL_PATH)
            
            print("✅ DL model loaded successfully with `load_model()`.")
            # DL_MODEL.summary()
        else:
            print(f"⚠️ ERROR: Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"🔥 FAILED to load DL model: {e}")
        DL_MODEL = None  # Fallback to avoid server crash

    yield
    print("🛑 Server shutting down...")



# Attach the lifespan manager to the FastAPI app
app = FastAPI(lifespan=lifespan)

METHOD_NAME_MAP = {
    "A": "DWT-DST-SVD-SS",
    "B": "DWT-DCT-SVD-SS",
    "C": "SWT-DST-QR-SS",
    "D": "SWT-DCT-QR-SS"
}
# Request model
class EmbedRequest(BaseModel):
    audio_url: str
    img_url: str
    method_identifier: str  # Expects "A", "B", "C", or "D" from Flutter
    subband: int
    bit: int
    alfass: float
    uploaded_by: str  # user UUID

class ExtractRequest(BaseModel):
    audio_url: str
    filename: str 
    uploaded_by: str  # user UUID


class ExtractDLRequest(BaseModel):
    audio_url: str
    filename: str 
    uploaded_by: str  # user UUID


class AttackRequest(BaseModel):
    audio_url: str
    original_filename: str
    attack_type: int
    attack_param: int
    uploaded_by: str  # user UUID



def resolve_unique_filename(bucket, folder, base_filename):
    name, ext = os.path.splitext(base_filename)
    attempt = 1
    candidate = base_filename

    while True:
        existing = supabase.storage.from_(bucket).list(path=folder)
        filenames = [item['name'] for item in existing if 'name' in item]

        if candidate not in filenames:
            return candidate  # unique filename found

        attempt += 1
        candidate = f"{name} ({attempt}){ext}"

def clear_temp_folders():
     for folder in ["temp_input", "temp_output"]:
        # DIAGNOSTIC: Announce which folder is being cleared
        print(f"--- [DIAGNOSTIC] Clearing temp folder: ./{folder} ---")
        files = glob.glob(f"{folder}/*")
        if not files:
            print(f"--- [DIAGNOSTIC] Folder is already empty. ---")
            continue
            
        for file in files:
            try:
                # DIAGNOSTIC: Announce which file is being deleted
                print(f"--- [DIAGNOSTIC] Deleting old file: {file} ---")
                os.remove(file)
            except Exception as e:
                print(f"⚠️ Failed to delete {file}: {e}")

def get_attack_name(attack_type: int, attack_param: int) -> str:
    # """Finds the human-readable attack name from the global ATTACK_LIST."""
    for attack_info in ATTACK_LIST:
        if attack_info[0] == [attack_type, attack_param]:
            return attack_info[1]
    return "Custom Attack" # Fallback name if not found in the list

# --- perform_dl_extraction function MODIFICATIONS ---
# In server.py, replace the existing perform_dl_extraction function with this one.

def perform_dl_extraction(audio_path: str, key_path: str, output_dir: str):
    if DL_MODEL is None:
        raise RuntimeError("DL Model is not loaded. Cannot perform extraction.")

    print(f"🕵️‍♂️ Processing audio='{os.path.basename(audio_path)}', key='{os.path.basename(key_path)}'")

    if not os.path.exists(key_path):
        raise FileNotFoundError(f"Key file not found at: {key_path}")
    
    mat_contents = scipy.io.loadmat(key_path)
    true_watermark_flat = mat_contents.get('wt_all')[0, 0].flatten().astype(int)
    
    PN_CODE_LENGTH = 1000
    STFT_DIMS = (1025, 256)
    WATERMARK_OUTPUT_DIMS = (32, 32)
    
    y, sr = librosa.load(audio_path, sr=None)
    
    M = int(mat_contents['M'][0][0])
    N = int(mat_contents['N'][0][0])
    B = int(mat_contents['B'][0][0])
    block_len = M * N * (B**2)
    all_block_nums = mat_contents['saved_original_indices'].flatten().astype(int)

    best_ber = float('inf')
    best_predicted_image = None
    best_predicted_attack = "N/A"
    best_block_num = -1

    print(f"🔬 Found {len(all_block_nums)} blocks. Starting evaluation...")

    for block_num in all_block_nums:
        start, end = (block_num - 1) * block_len, block_num * block_len
        audio_block = y[start:end]

        if audio_block.size == 0:
            print(f"  - WARNING: Block {block_num} is empty. Skipping.")
            continue
        
        stft_result = librosa.stft(audio_block)
        magnitude = np.abs(stft_result)
        phase = np.angle(stft_result)
        two_channel_stft = np.stack([magnitude / (np.max(magnitude) + 1e-6), phase], axis=-1)
        
        time_steps = STFT_DIMS[1]
        if two_channel_stft.shape[1] < time_steps:
            pad_width = time_steps - two_channel_stft.shape[1]
            two_channel_stft = np.pad(two_channel_stft, ((0, 0), (0, pad_width), (0, 0)))
        else:
            two_channel_stft = two_channel_stft[:, :time_steps, :]
        
        pn0 = mat_contents['pn0_all'][0, 0].flatten()
        pn1 = mat_contents['pn1_all'][0, 0].flatten()
        
        pn0 = np.pad(pn0, (0, PN_CODE_LENGTH - len(pn0))) if len(pn0) < PN_CODE_LENGTH else pn0[:PN_CODE_LENGTH]
        pn1 = np.pad(pn1, (0, PN_CODE_LENGTH - len(pn1))) if len(pn1) < PN_CODE_LENGTH else pn1[:PN_CODE_LENGTH]
        
        input_stft = np.expand_dims(two_channel_stft, axis=0)
        input_pn0 = np.expand_dims(pn0, axis=0)
        input_pn1 = np.expand_dims(pn1, axis=0)
        
        predicted_watermark_output, predicted_attack_probs = DL_MODEL.predict([input_stft, input_pn0, input_pn1], verbose=0)
        
        predicted_image = predicted_watermark_output[0].squeeze()
        predicted_binary_flat = (predicted_image.flatten() > 0.5).astype(int)
        
        ber = np.not_equal(predicted_binary_flat, true_watermark_flat).sum() / true_watermark_flat.size
        
        if ber < best_ber:
            best_ber = ber
            raw_image = (predicted_binary_flat.reshape(WATERMARK_OUTPUT_DIMS) * 255).astype(np.uint8)
            rotated_image = np.rot90(raw_image, k=-1)
            corrected_image = np.fliplr(rotated_image)
            best_predicted_image = corrected_image
            
            pred_attack_id = np.argmax(predicted_attack_probs[0])
            best_predicted_attack = id_to_attack_name.get(pred_attack_id, "Unknown Attack")
            best_block_num = block_num

    if best_predicted_image is None:
        raise ValueError("Could not process any audio blocks to generate a watermark image.")

    print(f"🏆 Selected Block {best_block_num} as the best result with BER: {best_ber:.4f}")
    
    output_img_filename = os.path.basename(audio_path).replace(".wav", "_extracted_dl.png")
    output_img_path = os.path.join(output_dir, output_img_filename)
    
    # The 'best_predicted_image' being saved here is now the corrected version
    plt.imsave(output_img_path, best_predicted_image, cmap='gray')
    
    return output_img_path, best_ber, best_predicted_attack
#-------------------------------------------------------- Ping Check -------------------------------------------------------
@app.get("/ping")
async def ping_server():
    # A simple endpoint to check if the server is alive and responding.
    # Get current time in Jakarta for the log message
    jakarta_time = datetime.now(ZoneInfo("Asia/Jakarta"))
    log_timestamp = jakarta_time.strftime("%Y-%m-%d %H:%M:%S") # Format for readability

    # Print a message to the server's console
    print(f"[{log_timestamp} Asia/Jakarta] Ping request received. Responding with 'pong'.")
    
    return {"status": "ok", "message": "pong"}
#`------------------------------------------------------- Embed Watermark -------------------------------------------------------`
@app.post("/embed")
async def embed_watermark(data: EmbedRequest, authorization: str | None = Header(default=None)):
    start_time = time.time()

    clear_temp_folders()  # Clear temp folders before processing
    jakarta_time = datetime.now(ZoneInfo("Asia/Jakarta"))

    # Formatter for database timestamp (YYYY-MM-DD HH:MM:SS+ZZ:ZZ)
    db_formatted_time = jakarta_time.strftime("%Y-%m-%d %H:%M:%S%z") #
    db_formatted_time = db_formatted_time[:-2] + ':' + db_formatted_time[-2:] #
    
    # Formatter for filename timestamp (YYMMDD_HHMM)
    filename_timestamp = jakarta_time.strftime("%y%m%d_%H%M") # YYMMDD_HHMM

    formatted_time = jakarta_time.strftime("%Y-%m-%d %H:%M:%S%z")
    formatted_time = formatted_time[:-2] + ':' + formatted_time[-2:]

    # ------------------- USER IMPERSONATION LOGIC -------------------
    if not authorization or not authorization.startswith("Bearer "):
        return {"status": "error", "message": "Unauthorized: Missing or invalid token"}
    
    access_token = authorization.split(" ")[1]

    # Create a new client authenticated AS THE USER
    user_supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    user_supabase.auth.set_session(access_token=access_token, refresh_token=access_token)
    # ------------------- END OF LOGIC -------------------

 
    print("Jakarta time:", formatted_time)
    print("Received request Embedding from:", data.uploaded_by)

    os.makedirs("temp_input", exist_ok=True)
    os.makedirs("temp_output", exist_ok=True)

    # Prepare filenames and paths
    audio_filename = os.path.basename(urlparse(data.audio_url).path)
    image_filename = os.path.basename(urlparse(data.img_url).path)

    input_audio_path = f"temp_input/{audio_filename}" #
    input_image_path = f"temp_input/{image_filename}" #
 
     # --- Create unique output filename with timestamp ---
    name_part, ext_part = os.path.splitext(audio_filename)
    unique_output_audio_filename = f"{name_part}-{filename_timestamp}{ext_part}"
    
    output_audio_path = f"temp_output/{unique_output_audio_filename}" #

    key_filename_derived = f"{name_part}-{filename_timestamp}_data.mat"
    snr_filename_derived = f"{name_part}-{filename_timestamp}_snr.txt"
    
    # Local paths on server where MATLAB executable saves these files
    output_key_path = f"temp_output/{key_filename_derived}"
    output_snr_path = f"temp_output/{snr_filename_derived}"

    try:
        audio_resp = requests.get(data.audio_url)
        audio_resp.raise_for_status()
        with open(input_audio_path, 'wb') as f:
            f.write(audio_resp.content)

        image_resp = requests.get(data.img_url)
        image_resp.raise_for_status()
        with open(input_image_path, 'wb') as f:
            f.write(image_resp.content)
                
        # Validate method_identifier (good practice)
        if data.method_identifier not in METHOD_NAME_MAP:
            return {"status": "error", "message": "Invalid method identifier provided."}

        # Run embedding
        result = subprocess.run([
            "embedding_production8v1.exe",
            input_audio_path,
            input_image_path,
            data.method_identifier,  # Pass "A", "B", "C", or "D" directly
            str(data.subband),
            str(data.bit),
            str(data.alfass),
            output_audio_path
        ], capture_output=True, timeout=300, text=True)

        print("📋 EMBEDDING STDOUT:\n", result.stdout)
        print("⚠️ EMBEDDING STDERR:\n", result.stderr)

        if result.returncode != 0:
            return {"status": "error", "message": result.stderr}


        # Check output files
        if not os.path.exists(output_audio_path) or not os.path.exists(output_key_path):
            return {"status": "error", "message": "Output files not generated properly."}

        # Read SNR
        try:
            with open(output_snr_path, 'r') as snr_file:
                snr_value = float(snr_file.read().strip())
        except:
            snr_value = None

        info = sf.info(output_audio_path)
        print(f"📤 Uploading file with sample rate: {info.samplerate}, frames: {info.frames}")

        # Upload to Supabase
        with open(output_audio_path, 'rb') as f:
            user_supabase.storage.from_('watermarked').upload(
                f"audios/{unique_output_audio_filename}",
                f,
                {"content-type": "audio/wav"}
            )
        with open(output_key_path, 'rb') as f:
            user_supabase.storage.from_('watermarked').upload(
                f"key/{os.path.basename(output_key_path)}",
                f,
                {"content-type": "application/octet-stream"}
            )
         # Get the full human-readable method name for database storage
        friendly_method_name = METHOD_NAME_MAP.get(data.method_identifier, "Unknown Method")


        audio_url = supabase.storage.from_('watermarked').get_public_url(f"audios/{unique_output_audio_filename}")
        key_url = supabase.storage.from_('watermarked').get_public_url(f"key/{os.path.basename(output_key_path)}")

        # Insert metadata
        supabase.table("audio_watermarked").insert({
            "filename": unique_output_audio_filename,
            "url": audio_url,
            "key_url": key_url,
            "method": friendly_method_name, # Store full name like "DWT-DST-SVD-SS"
            "bit": data.bit,
            "subband": data.subband,
            "alfass": data.alfass,
            "snr": snr_value,
            "attack": "Clean", 
            "source": None,     
            "uploaded_by": data.uploaded_by,
            "uploaded_at": formatted_time
        }).execute()

        end_time = time.time()
        duration = end_time - start_time
        print(f"⏱️ Embed process completed in {duration:.2f} seconds.")

        # CLEAN UP
        try:
            os.remove(input_audio_path)
            os.remove(input_image_path)
            os.remove(output_audio_path)
            os.remove(output_key_path)
            os.remove(output_snr_path)
        except Exception as e:
            print("⚠️ Cleanup warning:", e)


        return {
            "status": "success",
            "audio_url": audio_url,
            "key_url": key_url,
            "snr": snr_value,
            "watermarked_filename": unique_output_audio_filename
        }
        

    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Embedding process timed out."}
    except requests.RequestException as e:
        return {"status": "error", "message": f"Download error: {str(e)}"}
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"⏱️ Embed process FAILED after {duration:.2f} seconds.")
        return {"status": "error", "message": f"Unexpected server error: {str(e)}"}
    
# ------------------------------------------------------- Attack Audio -----------------------------------------------------------
# IN server.py, REPLACE THE ENTIRE /attack FUNCTION WITH THIS

@app.post("/attack")
async def apply_attack(data: AttackRequest, authorization: str | None = Header(default=None)):
    start_time = time.time()
    clear_temp_folders()
    jakarta_time = datetime.now(ZoneInfo("Asia/Jakarta"))
    db_formatted_time = jakarta_time.strftime("%Y-%m-%d %H:%M:%S%z")
    db_formatted_time = db_formatted_time[:-2] + ':' + db_formatted_time[-2:]

    # --- USER AUTHENTICATION ---
    if not authorization or not authorization.startswith("Bearer "):
        return {"status": "error", "message": "Unauthorized: Missing or invalid token"}
    access_token = authorization.split(" ")[1]
    user_supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    user_supabase.auth.set_session(access_token=access_token, refresh_token=access_token)
    
    print(f"Received request to ATTACK file: {data.original_filename}")
    print(f"Attack Type: {data.attack_type}, Param: {data.attack_param}")

    os.makedirs("temp_input", exist_ok=True)
    os.makedirs("temp_output", exist_ok=True)

    # --- Define file paths ---
    name_part, ext_part = os.path.splitext(data.original_filename)
    input_audio_path = f"temp_input/{data.original_filename}"
    key_filename = f"{name_part}_data.mat"
    input_key_path = f"temp_input/{key_filename}"
    
    attacked_audio_filename = f"{name_part}_att.wav"
    attacked_key_filename = f"{name_part}_att_data.mat"
    
    output_audio_path = f"temp_input/{attacked_audio_filename}"
    output_key_path = f"temp_input/{attacked_key_filename}"

    try:
        # --- START OF THE FIX: Check if attacked file already exists ---
        print(f"Checking for existing attacked file: {attacked_audio_filename}")
        existing_file_check = supabase.table("audio_watermarked").select("filename").eq("filename", attacked_audio_filename).execute()
        
        if existing_file_check.data:
            print("Conflict: Attacked file already exists in the database.")
            # Return a 409 Conflict status code
            return JSONResponse(
                status_code=409,
                content={"status": "error", "message": "This audio has already been attacked. An attacked audio can only be generated once."}
            )
        # --- END OF THE FIX ---

        # --- 1. Fetch original record to get URLs ---
        source_rows = supabase.table("audio_watermarked").select("*").eq("filename", data.original_filename).execute()
        if not source_rows.data:
            return {"status": "error", "message": "Original audio record not found in database."}
        source_data = source_rows.data[0]

        # --- 2. Download original audio AND key file ---
        audio_resp = requests.get(data.audio_url)
        audio_resp.raise_for_status()
        with open(input_audio_path, 'wb') as f:
            f.write(audio_resp.content)
        print(f"Downloaded audio file: {data.original_filename}")

        key_url = source_data.get('key_url')
        if not key_url:
            return {"status": "error", "message": "Key URL not found for the original audio."}
            
        key_resp = requests.get(key_url)
        key_resp.raise_for_status()
        with open(input_key_path, 'wb') as f:
            f.write(key_resp.content)
        print(f"Downloaded key file: {key_filename}")

        # --- 3. Run the attack executable ---
        print("Calling attack_production.exe...")
        result = subprocess.run([
            "attack_production.exe",
            str(data.attack_type),
            str(data.attack_param),
            input_audio_path
        ], capture_output=True, timeout=300, text=True)

        print("📋 ATTACK STDOUT:\n", result.stdout)
        print("⚠️ ATTACK STDERR:\n", result.stderr)

        if result.returncode != 0:
            return {"status": "error", "message": result.stderr or "Attack executable failed."}

        # --- 4. Verify and Upload new files ---
        if not os.path.exists(output_audio_path) or not os.path.exists(output_key_path):
            return {"status": "error", "message": "Attacked output files were not generated by the executable."}

        with open(output_audio_path, 'rb') as f:
            user_supabase.storage.from_('watermarked').upload(
                f"audios/{attacked_audio_filename}", f, {"content-type": "audio/wav"}
            )
        with open(output_key_path, 'rb') as f:
            user_supabase.storage.from_('watermarked').upload(
                f"key/{attacked_key_filename}", f, {"content-type": "application/octet-stream"}
            )

        # --- 5. Insert new record into database ---
        attacked_audio_url = supabase.storage.from_('watermarked').get_public_url(f"audios/{attacked_audio_filename}")
        attacked_key_url = supabase.storage.from_('watermarked').get_public_url(f"key/{attacked_key_filename}")
        attack_name = get_attack_name(data.attack_type, data.attack_param)

        supabase.table("audio_watermarked").insert({
            "filename": attacked_audio_filename,
            "url": attacked_audio_url,
            "key_url": attacked_key_url,
            "method": source_data.get('method'),
            "bit": source_data.get('bit'),
            "subband": source_data.get('subband'),
            "alfass": source_data.get('alfass'),
            "snr": None,
            "attack": attack_name,
            "source": data.original_filename,
            "uploaded_by": data.uploaded_by,
            "uploaded_at": db_formatted_time
        }).execute()
        
        print(f"Successfully created and logged attacked file: {attacked_audio_filename}")

        end_time = time.time()
        duration = end_time - start_time
        print(f"⏱️ Attack process completed in {duration:.2f} seconds.")

        return {
            "status": "success",
            "message": f"Attack '{attack_name}' applied successfully.",
            "attacked_audio_url": attacked_audio_url,
            "attacked_filename": attacked_audio_filename
        }

    except Exception as e:
        print(f"🔥 An error occurred during attack process: {e}")
        return {"status": "error", "message": f"An unexpected server error occurred: {str(e)}"}
    
# ------------------------------------------------------- Extract Watermark -------------------------------------------------------
@app.post("/extract")
async def extract_watermark(data: ExtractRequest):
    start_time = time.time()

    clear_temp_folders()
    print("Received request Extraction ")

    os.makedirs("temp_input", exist_ok=True)
    os.makedirs("temp_output", exist_ok=True)

    audio_filename = data.filename
    audio_path = f"temp_input/{audio_filename}"
    key_path = audio_path.replace(".wav", "_data.mat")

    # Query key_url from audio_watermarked table
    rows = supabase.table("audio_watermarked").select("key_url").eq("filename", data.filename).execute()
    if not rows.data:
        return {"status": "error", "message": "Key not found for this audio"}
    key_url = rows.data[0]["key_url"]

    # Download audio and key
    audio_resp = requests.get(data.audio_url)
    audio_resp.raise_for_status()
    with open(audio_path, 'wb') as f:
        f.write(audio_resp.content)

    key_resp = requests.get(key_url)
    key_resp.raise_for_status()
    with open(key_path, 'wb') as f:
        f.write(key_resp.content)

    print("✅ Extract - downloaded file size:", os.path.getsize(audio_path))
    print("✅ Checking downloaded audio file:", audio_path)

    try:
        info = sf.info(audio_path)
        print("📊 Audio info:")
        print(f" - Frames: {info.frames}")
        print(f" - Sample rate: {info.samplerate}")
        print(f" - Channels: {info.channels}")
        print(f" - Duration: {info.duration:.2f} sec")
    except Exception as e:
        print("⚠️ Failed to read audio info:", str(e))

    time.sleep(0.5)

    # Run the extraction executable
    result = subprocess.run([
        "extraction_production6.exe",
        audio_path
    ], capture_output=True, text=True)

    # ✅ Handle result output safely
    stdout_text = result.stdout or ""
    stderr_text = result.stderr or ""

    print("📋 EXTRACTION STDOUT:\n", stdout_text)
    print("⚠️ EXTRACTION STDERR:\n", stderr_text)

    if result.returncode != 0:
        return {"status": "error", "message": stderr_text}

    output_img_path = audio_path.replace(".wav", "_extracted_watermark.png")
    if not os.path.exists(output_img_path):
        return {"status": "error", "message": "Extracted image not found"}

    image_filename = resolve_unique_filename("watermarked", "images", os.path.basename(output_img_path))

    with open(output_img_path, 'rb') as f:
        supabase.storage.from_('watermarked').upload(
            f"images/{image_filename}",
            f,
            {"content-type": "image/png"}
        )

    watermark_url = supabase.storage.from_('watermarked').get_public_url(f"images/{image_filename}")

    # ✅ Extract BER (robust pattern match)
    ber_value = None
    ber_match = re.search(r'BER\s*=\s*([0-9.]+)', stdout_text)
    if ber_match:
        try:
            ber_value = float(ber_match.group(1))
        except Exception as e:
            print("Failed to parse BER:", str(e))

    timestamp = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S%z")
    timestamp = timestamp[:-2] + ':' + timestamp[-2:]

    supabase.table("image_extracted").insert({
        "filename": image_filename,
        "url": watermark_url,
        "source_audio": data.filename,
        "ber": ber_value,
        "uploaded_at": timestamp,
        "uploaded_by": data.uploaded_by
    }).execute()

    end_time = time.time()
    duration = end_time - start_time
    print(f"⏱️ Normal Extract process completed in {duration:.2f} seconds.")

    return {
        "status": "success",
        "watermark_url": watermark_url,
        "ber": ber_value
    }

# ------------------------------------------------------- Extract DL Watermark -------------------------------------------------------
@app.post("/extract-dl")
async def extract_watermark_dl(data: ExtractDLRequest):
    start_time = time.time()
    
    clear_temp_folders()
    print(f"\n--- [REQUEST START] Received request for DL Extraction on file: {data.filename} ---")

    temp_input_dir = "temp_input"
    temp_output_dir = "temp_output"
    os.makedirs(temp_input_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)

    audio_filename = data.filename
    input_audio_path = os.path.join(temp_input_dir, audio_filename)
    key_filename = audio_filename.replace(".wav", "_data.mat")
    input_key_path = os.path.join(temp_input_dir, key_filename)

    print(f"--- [DIAGNOSTIC] Input audio path set to: {input_audio_path} ---")
    print(f"--- [DIAGNOSTIC] Input key path set to:   {input_key_path} ---")

    rows = supabase.table("audio_watermarked").select("key_url").eq("filename", data.filename).execute()
    if not rows.data:
        return {"status": "error", "message": f"Key not found for audio file: {data.filename}"}
    key_url = rows.data[0]["key_url"]

    try:
        # Download the audio file
        audio_resp = requests.get(data.audio_url)
        audio_resp.raise_for_status()
        with open(input_audio_path, 'wb') as f:
            f.write(audio_resp.content)
        audio_size = os.path.getsize(input_audio_path)
        print(f"--- [DIAGNOSTIC] DOWNLOADED audio '{audio_filename}' | Size: {audio_size} bytes ---")

        # Download the key file
        key_resp = requests.get(key_url)
        key_resp.raise_for_status()
        with open(input_key_path, 'wb') as f:
            f.write(key_resp.content)
        key_size = os.path.getsize(input_key_path)
        print(f"--- [DIAGNOSTIC] DOWNLOADED key '{key_filename}' | Size: {key_size} bytes ---")

        # Call the corrected extraction function
        extracted_img_path, ber_value, predicted_attack = perform_dl_extraction(
            audio_path=input_audio_path,
            key_path=input_key_path,
            output_dir=temp_output_dir
        )
        
        print(f"🧠 DL Model Predicted Attack: {predicted_attack}")
        
        if not os.path.exists(extracted_img_path):
            return {"status": "error", "message": "DL extraction failed to produce an image file."}

        image_filename_unique = resolve_unique_filename("watermarked", "images", os.path.basename(extracted_img_path))
        
        with open(extracted_img_path, 'rb') as f:
            supabase.storage.from_('watermarked').upload(
                f"images/{image_filename_unique}", f, {"content-type": "image/png"}
            )

        watermark_url = supabase.storage.from_('watermarked').get_public_url(f"images/{image_filename_unique}")
        
        timestamp = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S%z")
        timestamp = timestamp[:-2] + ':' + timestamp[-2:]

        supabase.table("image_extracted").insert({
            "filename": image_filename_unique, 
            "url": watermark_url, 
            "source_audio": data.filename,
            "ber": ber_value, 
            "uploaded_at": timestamp, 
            "uploaded_by": data.uploaded_by
        }).execute()

        end_time = time.time()
        # FIX: Calculate duration before using it.
        duration = end_time - start_time
        print(f"⏱️ DL Extraction process completed in {duration:.2f} seconds.")

        return {
            "status": "success", 
            "watermark_url": watermark_url, 
            "ber": ber_value
        }

    except Exception as e:
        end_time = time.time()
        # FIX: Calculate duration before using it in the error log.
        duration = end_time - start_time
        print(f"⏱️ DL Extraction process FAILED after {duration:.2f} seconds.")
        
        print(f"🔥 An error occurred during DL extraction: {e}")
        return {"status": "error", "message": f"An unexpected server error occurred: {str(e)}"}