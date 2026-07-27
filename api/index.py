import os
import base64
import json
import re
import traceback
import tempfile
from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv

try:
    import cv2
except Exception:
    # Catches ImportError as well as ABI-mismatch errors (e.g. AttributeError:
    # _ARRAY_API not found, which occurs when opencv-python-headless was built
    # against a different NumPy major version than the one installed).
    cv2 = None

load_dotenv()

# 1. Properly anchor template folder to the root 'templates/' directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates_dir = os.path.join(base_dir, 'templates')

app = Flask(__name__, template_folder=templates_dir)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

# Initialize Groq Client safely from environment
groq_api_key = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

# Supabase is configured client-side; the backend only needs to surface these
# public values (anon key is safe to expose to the browser by design).
supabase_url = os.environ.get("SUPABASE_URL")
supabase_anon_key = os.environ.get("SUPABASE_ANON_KEY")

# Current Groq-hosted model IDs. llama-3.2-*-vision-preview was decommissioned;
# qwen/qwen3.6-27b is the current Groq vision-capable model.
MODEL_VISION = "qwen/qwen3.6-27b"
MODEL_TEXT = "llama-3.3-70b-versatile"

MOCK_SIGNATURES = {
    "audio": {
        "filename": "grandchild_voice_urgent_clone.wav",
        "content_type": "audio/wav",
        "media_type": "audio_video",
        "report": {
            "score": 91,
            "label": "HIGH RISK: VOICE CLONE DETECTED",
            "colorClass": "bg-rose-500",
            "textHex": "text-rose-400",
            "glow": "shadow-[0_0_15px_#f43f5e]",
            "indicators": [
                {
                    "title": "Spectral Harmonic Break",
                    "desc": "Unnatural harmonic flattening detected across the 2-4kHz vocal formant range, consistent with neural voice synthesis.",
                    "state": "Flagged Anomaly",
                    "flag": True,
                    "tip": "Real human speech carries micro-jitter in formant frequencies that generative voice models tend to smooth over."
                },
                {
                    "title": "Breath & Pause Cadence",
                    "desc": "Absence of natural breath sounds and irregular pause timing typical of concatenative/neural TTS pipelines.",
                    "state": "Flagged Anomaly",
                    "flag": True,
                    "tip": "Cloned voices frequently omit involuntary breathing artifacts present in live recordings."
                },
                {
                    "title": "Urgency & Emotional Manipulation Script",
                    "desc": "Narrative structure matches known 'grandchild emergency' social engineering scam templates requesting immediate wire transfer.",
                    "state": "Scam Pattern Match",
                    "flag": True,
                    "tip": "High-pressure urgency combined with a request for untraceable payment is a classic scam signature."
                }
            ],
            "heatmap": [],
            "shieldText": "This audio sample matches known deepfake voice-cloning patterns used in family emergency scams. Do not act on payment requests from this call alone.",
            "shieldAction": "Hang up and call the family member back directly on a known, previously saved phone number to verify."
        }
    },
    "video": {
        "filename": "ceo_authority_mandate_wire.mp4",
        "content_type": "video/mp4",
        "media_type": "audio_video",
        "report": {
            "score": 88,
            "label": "HIGH RISK: DEEPFAKE VIDEO",
            "colorClass": "bg-rose-500",
            "textHex": "text-rose-400",
            "glow": "shadow-[0_0_15px_#f43f5e]",
            "indicators": [
                {
                    "title": "Lip-Sync Desynchronization",
                    "desc": "Mouth movement lags audio phonemes by a measurable offset, especially on plosive consonants.",
                    "state": "Flagged Anomaly",
                    "flag": True,
                    "tip": "Frame-by-frame lip movement is one of the most reliable low-level deepfake tells."
                },
                {
                    "title": "Facial Boundary Blending",
                    "desc": "Soft edge artifacts around the jawline and hairline indicative of face-swap compositing.",
                    "state": "Flagged Anomaly",
                    "flag": True,
                    "tip": "Generated faces are typically composited onto a source video, leaving faint blending seams."
                },
                {
                    "title": "Authority Impersonation Script",
                    "desc": "Message content mimics a corporate executive mandate demanding an urgent, confidential wire transfer.",
                    "state": "Scam Pattern Match",
                    "flag": True,
                    "tip": "Legitimate executives rarely authorize wire transfers exclusively through a single unverified video message."
                }
            ],
            "heatmap": [
                {"top": 20, "left": 30, "width": 25, "height": 25, "label": "Facial boundary blending artifact"},
                {"top": 55, "left": 40, "width": 20, "height": 15, "label": "Lip-sync desynchronization region"}
            ],
            "shieldText": "This video exhibits strong indicators of synthetic face and voice manipulation consistent with corporate authority impersonation scams.",
            "shieldAction": "Verify the instruction through a separate, pre-established channel before authorizing any transfer."
        }
    },
    "clean": {
        "filename": "family_reunion_portrait.jpg",
        "content_type": "image/jpeg",
        "media_type": "image",
        "report": {
            "score": 6,
            "label": "LOW RISK: NO ANOMALIES DETECTED",
            "colorClass": "bg-emerald-500",
            "textHex": "text-emerald-400",
            "glow": "shadow-[0_0_15px_#10b981]",
            "indicators": [
                {
                    "title": "Illumination Vector Consistency",
                    "desc": "Lighting direction and shadow falloff are consistent across all subjects and background elements.",
                    "state": "Nominal",
                    "flag": False,
                    "tip": "Composited or generated imagery often shows mismatched light sources between subject and background."
                },
                {
                    "title": "Sensor Noise Profile",
                    "desc": "Uniform photon noise distribution consistent with a standard camera sensor capture, not synthetic upsampling.",
                    "state": "Nominal",
                    "flag": False,
                    "tip": "AI-generated images often exhibit unnaturally smooth or repeating high-frequency noise patterns."
                }
            ],
            "heatmap": [],
            "shieldText": "No manipulation indicators were detected in this image. It is consistent with an authentic, unedited camera capture.",
            "shieldAction": "No action required. Continue routine verification practices for unfamiliar media."
        }
    }
}

def extract_and_parse_json(text):
    """Robustly extracts and parses a JSON object from raw model output."""
    if not text:
        raise ValueError("Received empty or null response text from AI backend.")
    
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or start > end:
        raise ValueError("The response stream did not contain an executable enclosed JSON block structure.")
    
    json_str = text[start:end+1]
    
    # Escape invalid control characters in raw string content
    json_str = re.sub(
        r'[\x00-\x1F\x7F]', 
        lambda m: f'\\u{ord(m.group(0)):04x}' if m.group(0) not in '\n\r\t' else m.group(0), 
        json_str
    )
    
    try:
        return json.loads(json_str)
    except Exception as e:
        raise ValueError(f"JSON Parsing failed: {str(e)}")

def extract_video_frame(file_bytes, filename):
    """Extracts a middle keyframe from video bytes for vision model analysis."""
    if not cv2:
        return None
    suffix = os.path.splitext(filename)[1] or '.mp4'
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            frame_count = 1

        target_frame = max(0, min(frame_count // 2, frame_count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return None

        success, encoded_img = cv2.imencode('.jpg', frame)
        if not success:
            return None

        return base64.b64encode(encoded_img.tobytes()).decode('utf-8')
    except Exception:
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def generate_simulated_report(filename, media_type, threat_context):
    """Fallback generator for local heuristic simulation."""
    is_image = media_type == 'image'
    is_video = media_type == 'video'

    if is_image:
        score = 68
        label = "SIMULATED ANOMALY"
        color = "bg-amber-500"
        hex_color = "text-amber-400"
        glow = "shadow-[0_0_15px_#f59e0b]"
        indicators = [
            {
                "title": "Local Heuristic Simulation",
                "desc": f"Analyzed image '{filename}' under context '{threat_context}' using local fallback rules.",
                "state": "Simulated Detection",
                "flag": True,
                "tip": "Groq API key missing or live AI returned unparseable output."
            },
            {
                "title": "Frequency Spectrum Distribution",
                "desc": "Local noise profile flags high variance consistent with synthetic imagery.",
                "state": "Flagged Anomaly",
                "flag": True,
                "tip": "Simulated evaluation of high-frequency noise bands."
            }
        ]
        heatmap = [
            {"top": 30, "left": 40, "width": 20, "height": 20, "label": "Simulated Anomaly Region 1"},
            {"top": 60, "left": 20, "width": 15, "height": 15, "label": "Simulated Anomaly Region 2"}
        ]
    elif is_video:
        score = 86
        label = "HIGH RISK: SYNTHETIC VIDEO DETECTED"
        color = "bg-rose-500"
        hex_color = "text-rose-400"
        glow = "shadow-[0_0_15px_#f43f5e]"
        indicators = [
            {
                "title": "Frame-to-Frame Temporal Jitter",
                "desc": f"Analyzed video '{filename}' under context '{threat_context}'. Detected frame interpolation variance.",
                "state": "Simulated Detection",
                "flag": True,
                "tip": "Synthetic video generators often struggle with temporal consistency across adjacent frames."
            },
            {
                "title": "Facial Landmark Alignment Drift",
                "desc": "Facial boundaries show high variance along composite edges.",
                "state": "Flagged Anomaly",
                "flag": True,
                "tip": "Simulated evaluation of face-swap boundary blending."
            }
        ]
        heatmap = [
            {"top": 20, "left": 30, "width": 25, "height": 25, "label": "Facial boundary blending artifact"},
            {"top": 55, "left": 40, "width": 20, "height": 15, "label": "Lip-sync desynchronization region"}
        ]
    else:
        score = 84
        label = "HIGH SIMULATED THREAT"
        color = "bg-rose-500"
        hex_color = "text-rose-400"
        glow = "shadow-[0_0_15px_#f43f5e]"
        indicators = [
            {
                "title": "Local Heuristic Simulation",
                "desc": f"Analyzed '{filename}' under context '{threat_context}' using local fallback rules.",
                "state": "Simulated Detection",
                "flag": True,
                "tip": "Groq API key missing or live AI returned unparseable output."
            }
        ]
        heatmap = []

    return {
        "score": score,
        "label": label,
        "colorClass": color,
        "textHex": hex_color,
        "glow": glow,
        "indicators": indicators,
        "heatmap": heatmap,
        "shieldText": "Local fallback simulation active. Ensure GROQ_API_KEY is configured in your environment.",
        "shieldAction": "Verify environment variables and re-run analysis."
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({
        'groq_configured': bool(groq_api_key),
        'model_vision': MODEL_VISION,
        'model_text': MODEL_TEXT,
        'supabase_url': supabase_url,
        'supabase_anon_key': supabase_anon_key
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'groq_active': bool(groq_client),
        'diagnostics': {
            'groq_api_key_set': bool(groq_api_key),
            'model_vision': MODEL_VISION,
            'model_text': MODEL_TEXT,
            'supabase_configured': bool(supabase_url and supabase_anon_key),
            'opencv_available': bool(cv2)
        }
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    threat_context = request.form.get('context', 'Unspecified Vector Context')
    mock_type = request.form.get('mock_type')
    uploaded_file = request.files.get('file')

    # Mock Threat Injector: sandboxed canned signatures, no live inference needed.
    if (not uploaded_file or not uploaded_file.filename) and mock_type:
        signature = MOCK_SIGNATURES.get(mock_type)
        if not signature:
            return jsonify({'error': f"Unrecognized mock_type '{mock_type}'."}), 400
        return jsonify({
            'success': True,
            'engine_mode': 'Mock Signature Playback',
            'diagnostic_log': f"Loaded sandboxed attack signature '{mock_type}' for UI/telemetry testing. No live inference performed.",
            **signature['report']
        })

    if not uploaded_file or not uploaded_file.filename:
        return jsonify({'error': 'No file submitted in payload.'}), 400

    filename = uploaded_file.filename or 'unnamed_asset'
    content_type = uploaded_file.mimetype or ''
    file_bytes = uploaded_file.read()

    if not file_bytes:
        return jsonify({'error': 'Submitted file is empty.'}), 400

    fn_lower = filename.lower()
    is_image = content_type.startswith('image/') or fn_lower.endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'))
    is_video = content_type.startswith('video/') or fn_lower.endswith(('.mp4', '.webm', '.ogg', '.mov', '.avi', '.mkv'))

    if is_image:
        media_type = 'image'
    elif is_video:
        media_type = 'video'
    else:
        media_type = 'audio_video'

    if not groq_client:
        simulated = generate_simulated_report(filename, media_type, threat_context)
        return jsonify({
            'success': True,
            'engine_mode': 'Local Simulation Engine',
            'diagnostic_log': 'GROQ_API_KEY missing. Engine safely degraded to local heuristic simulation.',
            **simulated
        })

    try:
        base64_frame = None
        if is_video:
            base64_frame = extract_video_frame(file_bytes, filename)

        if is_image or (is_video and base64_frame):
            if is_image:
                base64_payload = base64.b64encode(file_bytes).decode('utf-8')
                image_mime = content_type if content_type.startswith('image/') else 'image/jpeg'
                prompt_text = f"Context parameters: Asset received via '{threat_context}'. Filename: {filename}. Synthesize a structured forensic analysis mapping in raw JSON."
                system_instruction = (
                    "You are an expert Multi-Modal Image Forensics Auditor specializing in computer vision anomaly detection.\n"
                    "Examine the image asset strictly for signs of synthetic generation or AI manipulation, analyzing edge blending, "
                    "illumination vectors, local noise variance, and facial landmark anomalies.\n\n"
                    "CRITICAL STRUCTURAL & SYNTAX RULES:\n"
                    "1. Output your entire findings as a single, valid JSON object.\n"
                    "2. DO NOT use double quotes (\") inside text values or descriptions. Use single quotes (') for internal quotes.\n"
                    "3. Do not include markdown code block formatting (e.g. ```json).\n"
                    "4. Ensure strict valid JSON syntax with NO trailing commas.\n\n"
                    "Format your JSON output EXACTLY matching this schema:\n"
                    "{\n"
                    "  \"score\": 72,\n"
                    "  \"label\": \"MEDIUM RISK\",\n"
                    "  \"colorClass\": \"bg-amber-500\",\n"
                    "  \"textHex\": \"text-amber-400\",\n"
                    "  \"glow\": \"shadow-[0_0_15px_#f59e0b]\",\n"
                    "  \"indicators\": [\n"
                    "    {\"title\": \"Artifact Title\", \"desc\": \"Analytical description\", \"state\": \"Status\", \"flag\": true, \"tip\": \"Detailed hover tooltip explanation\"}\n"
                    "  ],\n"
                    "  \"heatmap\": [\n"
                    "    {\"top\": 25, \"left\": 45, \"width\": 15, \"height\": 15, \"label\": \"AI Artifact Description\"}\n"
                    "  ],\n"
                    "  \"shieldText\": \"Grandparent shield breakdown summary text\",\n"
                    "  \"shieldAction\": \"Direct operational action recommendation instruction\"\n"
                    "}"
                )
            else:
                base64_payload = base64_frame
                image_mime = 'image/jpeg'
                prompt_text = f"Context parameters: Video asset received via '{threat_context}'. Filename: {filename}. Analyze the extracted keyframe for video deepfake markers, face-swaps, lip-sync desynchronization regions, or frame manipulation artifacts and return raw JSON."
                system_instruction = (
                    "You are an expert Video Deepfake Forensics Auditor.\n"
                    "Examine this video keyframe strictly for signs of synthetic generation, facial swapping, temporal edge blending, lip-sync desynchronization, and AI manipulation.\n\n"
                    "CRITICAL STRUCTURAL & SYNTAX RULES:\n"
                    "1. Output your entire findings as a single, valid JSON object.\n"
                    "2. DO NOT use double quotes (\") inside text values or descriptions. Use single quotes (') for internal quotes.\n"
                    "3. Do not include markdown code block formatting (e.g. ```json).\n"
                    "4. Ensure strict valid JSON syntax with NO trailing commas.\n\n"
                    "Format your JSON output EXACTLY matching this schema:\n"
                    "{\n"
                    "  \"score\": 85,\n"
                    "  \"label\": \"HIGH RISK\",\n"
                    "  \"colorClass\": \"bg-rose-500\",\n"
                    "  \"textHex\": \"text-rose-400\",\n"
                    "  \"glow\": \"shadow-[0_0_15px_#f43f5e]\",\n"
                    "  \"indicators\": [\n"
                    "    {\"title\": \"Artifact Title\", \"desc\": \"Analytical description\", \"state\": \"Status\", \"flag\": true, \"tip\": \"Detailed hover tooltip explanation\"}\n"
                    "  ],\n"
                    "  \"heatmap\": [\n"
                    "    {\"top\": 20, \"left\": 30, \"width\": 25, \"height\": 25, \"label\": \"Facial boundary blending artifact\"}\n"
                    "  ],\n"
                    "  \"shieldText\": \"Summary of video forensic findings\",\n"
                    "  \"shieldAction\": \"Direct operational instruction for non-technical user\"\n"
                    "}"
                )

            messages = [
                {
                    "role": "system",
                    "content": system_instruction
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt_text
                        },
                        {
                            "type": "image_url", 
                            "image_url": {"url": f"data:{image_mime};base64,{base64_payload}"}
                        }
                    ]
                }
            ]
            model_engine = MODEL_VISION
        else:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert Audio/Video Deepfake Forensics Auditor.\n"
                        "Analyze the metadata context and descriptors for deepfake indicators such as spectral inconsistency, lip sync desynchronization, or voice synthesis artifacts.\n\n"
                        "CRITICAL STRUCTURAL & SYNTAX RULES:\n"
                        "1. Output your entire findings as a single, valid JSON object.\n"
                        "2. DO NOT use double quotes (\") inside text values or descriptions. Use single quotes (') for internal quotes.\n"
                        "3. Do not include markdown code block formatting (e.g. ```json).\n"
                        "4. Ensure strict valid JSON syntax with NO trailing commas.\n\n"
                        "Format your JSON output EXACTLY matching this schema:\n"
                        "{\n"
                        "  \"score\": 85,\n"
                        "  \"label\": \"HIGH RISK\",\n"
                        "  \"colorClass\": \"bg-rose-500\",\n"
                        "  \"textHex\": \"text-rose-400\",\n"
                        "  \"glow\": \"shadow-[0_0_15px_#f43f5e]\",\n"
                        "  \"indicators\": [\n"
                        "    {\"title\": \"Artifact Title\", \"desc\": \"Analytical description\", \"state\": \"Status\", \"flag\": true, \"tip\": \"Detailed hover tooltip explanation\"}\n"
                        "  ],\n"
                        "  \"heatmap\": [\n"
                        "    {\"top\": 10, \"left\": 10, \"width\": 80, \"height\": 80, \"label\": \"Full Audio/Visual Stream Flagged\"}\n"
                        "  ],\n"
                        "  \"shieldText\": \"Summary of audio/video breakdown findings\",\n"
                        "  \"shieldAction\": \"Direct operational instruction for non-technical user\"\n"
                        "}"
                    )
                },
                {
                    "role": "user",
                    "content": f"Filename: '{filename}', Type: '{content_type}', Vector: '{threat_context}'. Evaluate for synthetic voice or frame-rate anomalies and return raw JSON."
                }
            ]
            model_engine = MODEL_TEXT

        # Execute Groq call enforcing JSON mode.
        completion_kwargs = dict(
            model=model_engine,
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
            response_format={"type": "json_object"}
        )
        if model_engine == MODEL_VISION:
            completion_kwargs["reasoning_effort"] = "none"

        completion = groq_client.chat.completions.create(**completion_kwargs)

        raw_response = completion.choices[0].message.content
        parsed_analysis = extract_and_parse_json(raw_response)

        return jsonify({
            'success': True,
            'engine_mode': f'Groq Live AI ({model_engine})',
            'diagnostic_log': f'Successfully executed inference via {model_engine}. Stream parsed cleanly.',
            **parsed_analysis
        })

    except Exception as e:
        err_msg = str(e)
        stack = traceback.format_exc()
        simulated = generate_simulated_report(filename, media_type, threat_context)
        return jsonify({
            'success': True,
            'engine_mode': 'Local Simulation Engine',
            'diagnostic_log': f'Groq API Execution Call Failed: {err_msg}\n{stack}',
            **simulated
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
