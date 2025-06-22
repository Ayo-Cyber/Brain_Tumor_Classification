from passlib.context import CryptContext
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ::%(levelname)s::%(name)s --> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(raw_password: str):
    return pwd_context.hash(raw_password)


def verify_hash(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def log_message(level: str, message: str):
    if level == "info":
        return logger.info(message)
    if level == "error":
        return logger.error(message)
    if level == "warning":
        return logger.warning(message)
