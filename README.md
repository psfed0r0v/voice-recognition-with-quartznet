# Voice Recognition with Quartznet

Used trained model QuartzNet15x5Base-En from this [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/models/nemospeechmodels)
and previously converted model to .onnx format for easier usage

Model inference docker container with Nvidia Triton in [triton](triton) module 

Docker container with FastApi client which processes incoming audio, converts it to tensors, sends it to inference
and generates the response text in [asr_client](asr_client) module

Python-script to run Telegram bot in [telegram_bot](telegram_bot) module
which translates english voice messages to text
(paste your API_TOKEN to test bot in [telegram_bot](telegram_bot/run.py))

For testing services locally execute scripts in [scripts](scripts) directory
```
./run_asr_client.sh
./run_triton_server.sh
./run_telegram_bot.sh
```
