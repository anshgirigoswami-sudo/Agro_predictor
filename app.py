from Main import app, load_persisted_models, models


if not models:
    load_persisted_models(evaluate=False)


if __name__ == '__main__':
    port = int(__import__('os').getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
