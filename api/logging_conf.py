import logging, os, json
from pythonjsonlogger import jsonlogger
from logstash_async.handler import AsynchronousLogstashHandler

def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # console JSON
    console = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)

    # logstash
    host = os.environ.get("LOGSTASH_HOST", "localhost")
    port = int(os.environ.get("LOGSTASH_PORT", "5044"))
    try:
        ls = AsynchronousLogstashHandler(host, port, database_path=None)
        logger.addHandler(ls)
    except Exception as e:
        logger.warning(f"Logstash handler not configured: {e}")
