import fastapi

import llm_benchmark_suite

app = fastapi.FastAPI()


@app.get("/")
async def root():
    return fastapi.Response(
        status_code=302, headers={"location": "https://thefastest.ai"}
    )


@app.route("/bench", methods=["GET", "POST"])
async def bench(req: fastapi.Request):
    text, content_type = await llm_benchmark_suite.run(req.query_params)
    return fastapi.Response(content=text, media_type=content_type)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # Run the app with uvicorn on port 8000
