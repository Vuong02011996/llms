# Run model llama 3.1 8b in local
# Viet api cho model llama vaof source co san.
    + /home/dxtech/Project/llm_local/tanika-ai-service/app/routers/llm_local_controller.py
    + api
        + generate-text
        + embedding

    + /home/dxtech/Project/llm_local/tanika-ai-service/app/core/singleton/loadModel.py
    + Run by docker not using conda:
        + docker compose exec ai_service bash (install new library)
        + docker compose logs ai_service -f 
        + `docker compose up -d ai_service`
        + Test api localhost:8002 | http://localhost:8002/docs#/
