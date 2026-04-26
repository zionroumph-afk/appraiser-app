from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import uvicorn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Moondream2 Vision Server")

MODEL_ID = "vikhyatk/moondream2"
REVISION  = "2024-07-23"  # stable revision that works with transformers 4.x+

print("⏳ Loading Moondream2 vision model...")
print("   First run downloads ~3.6GB — please wait...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    revision=REVISION,
    torch_dtype=torch.float32,
    device_map={"": "cpu"}
)
model.eval()

print("✅ Moondream2 ready!")

@app.post("/describe")
async def describe(image: UploadFile = File(...)):
    try:
        content = await image.read()
        img = Image.open(BytesIO(content)).convert("RGB")

        enc = model.encode_image(img)

        description = model.answer_question(
            enc,
            "Describe this image in detail, including what objects, characters, colors, and any visible text you can see.",
            tokenizer
        ).strip()

        print(f"✅ Described: {description[:120]}...")
        return JSONResponse({"description": description})

    except Exception as e:
        print(f"❌ Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
async def home():
    return {"status": "ok", "message": "Moondream2 running on port 5001"}

if __name__ == "__main__":
    print("🚀 Starting Moondream2 vision server on http://127.0.0.1:5001")
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="warning")