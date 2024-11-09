from utils import init_logging

LOG_PATH = "latest.log"
logger = init_logging(__name__, log_path=LOG_PATH)


def main() -> None:
    logger.info("Hello, World!")
    logger.debug("This is a debug message.")


if __name__ == "__main__":
    main()
