from fastapi import FastAPI, HTTPException, Depends

from model.model import get_response

app = FastAPI()


@app.get('/ping')
def health_check():
    """
    Function to check health status of endpoint.
    """
    return {"status": "healthy"}


@app.post('/predict')
def serve_model(data: dict):
    try:
        user_input = data.get('input')
        if not user_input:
            raise HTTPException(status_code=400, detail="Input is missing")

        response = get_response(user_input)

        return {"predictions": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)
