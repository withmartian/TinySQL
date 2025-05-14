import random
import string
import uuid
from datetime import date


def random_numeric_value():
    return str(random.randint(1, 100))


def random_text_value():
    char = random.choice(string.ascii_lowercase)
    return "'%" + char + "%'"


def random_date_value():
    year = random.randint(2000, 2023)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"'{year:04d}-{month:02d}-{day:02d}'"


def random_boolean_value():
    return random.choice(["TRUE", "FALSE"])


def random_uuid_value():
    return "'" + str(uuid.uuid4()) + "'"


def random_blob_value():
    return "x'" + ''.join(random.choices('0123456789ABCDEF', k=8)) + "'"


def random_json_value():
    words = ["foo", "bar", "baz", "qux", "test"]
    key = random.choice(words)
    value = random.choice(words)
    return "'{\"" + key + "\": \"" + value + "\"}'"
