from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
from PIL import Image
from io import BytesIO
import uvicorn

app = FastAPI(title="Offline Vision Server")

# OCR engine (local, CPU)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def simple_caption(img: Image.Image) -> str:
    """
    A tiny offline captioner.
    Not fancy, but works on CPU with no Torch.
    """
    w, h = img.size

    if w < 50 or h < 50:
        return "A very small image."

    if w > h:
        shape = "a wide image"
    elif h > w:
        shape = "a tall image"
    else:
        shape = "a square image"

    return f"An image that appears to be {shape} with visible content."

@app.post("/describe")
async def describe(image: UploadFile = File(...)):
    try:
        content = await image.read()
        img = Image.open(BytesIO(content)).convert("RGB")

        # Caption (very lightweight)
        caption = simple_caption(img)

        # OCR text extraction
        ocr_result = ocr.ocr(content, cls=True)
        extracted_text = []

        if ocr_result:
            for line in ocr_result:
                for box in line:
                    extracted_text.append(box[1][0])

        text = " ".join(extracted_text).strip()

        final_description = caption
        if text:
            final_description += f" It contains text: '{text}'."

        return JSONResponse({"description": final_description})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
async def home():
    return {"status": "ok", "message": "Offline vision server running"}

if __name__ == "__main__":
    print("🚀 Starting offline vision server on http://127.0.0.1:5001")
    uvicorn.run(app, host="0.0.0.0", port=5001)
