import io
from fastapi import FastAPI, Request, File, UploadFile, Depends, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import base64
from PIL import Image
from fastapi.responses import RedirectResponse

from service.object_detection import ObjectDetectionService

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def object_detection(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/object-detection-histogram/", response_class=HTMLResponse)
async def object_detection_histogram(
    request: Request,
    file: UploadFile = File(...),
    object_detection_service: ObjectDetectionService = Depends(),
):
    try:
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    except Exception:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    finally:
        file.file.close()

    prediction_score_threshold = 0.75
    predictions = object_detection_service.get_predictions_over_threshold(
        image,
        prediction_score_threshold,
    )

    predictions_context = []
    for prediction in predictions:
        hist, _ = object_detection_service.get_prediction_histogram(image, prediction)
        predictions_context.append(
            {
                "score": prediction.score,
                "hist": hist.tolist(),
            }
        )

    return templates.TemplateResponse(
        "histogram.html",
        {
            "request": request,
            "image": image_base64,
            "prediction_score_threshold": prediction_score_threshold,
            "predictions": predictions_context,
            "histogram_bins": list(range(256)),
        },
    )
