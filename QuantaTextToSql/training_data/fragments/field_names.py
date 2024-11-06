import random
from typing import Dict, List, Tuple
from functools import lru_cache
from .models import TableField, SelectField


@lru_cache(maxsize=1)
def get_sql_field_names_and_types() -> Dict[str, List[str]]:
    """Cache the combined dictionary since it's static and expensive to compute."""
    return combine_column_types(
        get_common_sql_columns_with_types_claude(),
        get_sql_column_names_and_types_chatgpt()
    )


@lru_cache(maxsize=1)
def get_field_names_and_types_list() -> List[Tuple[str, List[str]]]:
    """Cache the conversion of dictionary to list for sampling."""
    return list(get_sql_field_names_and_types().items())


def get_sql_table_fields(table_name: str, N: int) -> List[TableField]:
    """Return N random SQL field names with an associated data type."""
    # Get the cached list of field names and their possible types
    field_names = get_field_names_and_types_list()
    
    # Sample N items from the list, in a random order
    selected_fields = random.sample(field_names, N)
    
    # For each selected field, choose one random type from its possible types
    answer = [TableField(name=name, type=random.choice(types)) for name, types in selected_fields]

    # If the table name is also a field name, change add "_field" to the field name
    for a_field in answer:
        if a_field.name == table_name:
            a_field.name += "_field"

    return answer


def get_sql_select_fields( table_fields: List[TableField], N: int, use_aggregates: bool) -> List[SelectField]:
    """Return N random SQL field names with an associated aggregate."""
    
    # Define which aggregates can be used with which data types
    aggregate_by_type = {
        "INTEGER": ["SUM", "AVG", "MIN", "MAX", "COUNT", ""],
        "BIGINT": ["SUM", "AVG", "MIN", "MAX", "COUNT", ""],
        "DECIMAL": ["SUM", "AVG", "MIN", "MAX", "COUNT", ""],
        "NUMERIC": ["SUM", "AVG", "MIN", "MAX", "COUNT", ""],
        "FLOAT": ["SUM", "AVG", "MIN", "MAX", "COUNT", ""],
        "DOUBLE": ["SUM", "AVG", "MIN", "MAX", "COUNT", ""],
        "VARCHAR": ["MIN", "MAX", "COUNT", ""],
        "CHAR": ["MIN", "MAX", "COUNT", ""],
        "TEXT": ["MIN", "MAX", "COUNT", ""],
        "DATE": ["MIN", "MAX", "COUNT", ""],
        "DATETIME": ["MIN", "MAX", "COUNT", ""],
        "TIMESTAMP": ["MIN", "MAX", "COUNT", ""],
        "BOOLEAN": ["COUNT", ""],
        "UUID": ["COUNT", ""],
        "BLOB": ["COUNT", ""],
        "JSON": ["COUNT", ""],
        "JSONB": ["COUNT", ""]
    }
    
    # Default to all types if type not found
    default_aggregates = ["COUNT", ""]
    
    selected_fields = []   
    for a_table_field in random.sample(table_fields, N):

        field_name = a_table_field.name
        field_type = a_table_field.type.upper()
        
        if use_aggregates:
            # Get valid aggregates for this field type
            valid_aggregates = aggregate_by_type.get(field_type, default_aggregates)
            agg = random.choice(valid_aggregates)
            selected_fields += [SelectField(name=field_name, aggregate=agg)]
        else:
            selected_fields += [SelectField(name=field_name, aggregate="")]

    return selected_fields


def get_field_names_count():
    return len(get_field_names_and_types_list())


def combine_column_types(dict1, dict2):
    combined = {}
    all_keys = set(dict1.keys()).union(dict2.keys())
    for key in all_keys:
        types1 = dict1.get(key, [])
        types2 = dict2.get(key, [])
        # Combine and remove duplicates while preserving order
        combined_types = list(dict.fromkeys(types1 + types2))
        combined[key] = combined_types
    return combined


def get_common_sql_columns_with_types_claude():
    """
    Returns a dictionary of common SQL column names and their frequently used data types.
    Each column name maps to a list of common SQL data types for that column.
    """
    return {
        # Primary Keys and IDs
        "id": ["INTEGER", "BIGINT", "SERIAL", "BIGSERIAL"],
        "uuid": ["UUID", "VARCHAR(36)", "CHAR(36)"],
        "guid": ["VARCHAR(36)", "CHAR(36)"],
        "external_id": ["VARCHAR(50)", "VARCHAR(100)"],
        "reference_id": ["VARCHAR(50)", "VARCHAR(100)"],
        "parent_id": ["INTEGER", "BIGINT"],
        "source_id": ["INTEGER", "BIGINT"],
        "target_id": ["INTEGER", "BIGINT"],
        
        # Common Prefixed IDs
        "user_id": ["INTEGER", "BIGINT"],
        "customer_id": ["INTEGER", "BIGINT"],
        "order_id": ["INTEGER", "BIGINT"],
        "product_id": ["INTEGER", "BIGINT"],
        "account_id": ["INTEGER", "BIGINT"],
        "session_id": ["VARCHAR(100)", "CHAR(32)"],
        "transaction_id": ["VARCHAR(50)", "VARCHAR(100)"],
        
        # Basic Information
        "name": ["VARCHAR(100)", "VARCHAR(255)"],
        "title": ["VARCHAR(100)", "VARCHAR(255)"],
        "description": ["TEXT", "VARCHAR(1000)", "MEDIUMTEXT"],
        "summary": ["TEXT", "VARCHAR(500)"],
        "details": ["TEXT", "JSONB", "JSON"],
        "content": ["TEXT", "MEDIUMTEXT", "LONGTEXT"],
        "notes": ["TEXT", "VARCHAR(1000)"],
        "comments": ["TEXT", "VARCHAR(1000)"],
        "type": ["VARCHAR(50)", "ENUM"],
        "category": ["VARCHAR(50)", "VARCHAR(100)"],
        "status": ["VARCHAR(20)", "ENUM", "SMALLINT"],
        "code": ["VARCHAR(50)", "CHAR(10)"],
        "slug": ["VARCHAR(100)", "VARCHAR(255)"],
        
        # Name Components
        "first_name": ["VARCHAR(50)", "VARCHAR(100)"],
        "last_name": ["VARCHAR(50)", "VARCHAR(100)"],
        "middle_name": ["VARCHAR(50)", "VARCHAR(100)"],
        "full_name": ["VARCHAR(150)", "VARCHAR(255)"],
        "display_name": ["VARCHAR(100)", "VARCHAR(255)"],
        "username": ["VARCHAR(50)", "VARCHAR(100)"],
        "nickname": ["VARCHAR(50)", "VARCHAR(100)"],
        
        # Contact Information
        "email": ["VARCHAR(100)", "VARCHAR(255)"],
        "phone": ["VARCHAR(20)", "VARCHAR(50)"],
        "mobile": ["VARCHAR(20)", "VARCHAR(50)"],
        "fax": ["VARCHAR(20)", "VARCHAR(50)"],
        "website": ["VARCHAR(255)", "TEXT"],
        "phone_number": ["VARCHAR(20)", "VARCHAR(50)"],
        
        # Address Components
        "address": ["VARCHAR(255)", "TEXT"],
        "address_line1": ["VARCHAR(255)"],
        "address_line2": ["VARCHAR(255)"],
        "street": ["VARCHAR(255)"],
        "city": ["VARCHAR(100)"],
        "state": ["VARCHAR(100)", "CHAR(2)"],
        "country": ["VARCHAR(100)", "CHAR(2)", "CHAR(3)"],
        "postal_code": ["VARCHAR(20)", "CHAR(5)", "CHAR(10)"],
        "zip_code": ["VARCHAR(20)", "CHAR(5)", "CHAR(10)"],
        "latitude": ["DECIMAL(10,8)", "FLOAT", "DOUBLE"],
        "longitude": ["DECIMAL(11,8)", "FLOAT", "DOUBLE"],
        
        # Dates and Times
        "created_at": ["TIMESTAMP", "DATETIME"],
        "updated_at": ["TIMESTAMP", "DATETIME"],
        "deleted_at": ["TIMESTAMP", "DATETIME"],
        "modified_at": ["TIMESTAMP", "DATETIME"],
        "timestamp": ["TIMESTAMP", "BIGINT"],
        "date": ["DATE"],
        "time": ["TIME"],
        "datetime": ["DATETIME", "TIMESTAMP"],
        "start_date": ["DATE", "DATETIME"],
        "end_date": ["DATE", "DATETIME"],
        "duration": ["INTEGER", "INTERVAL"],
        "birthday": ["DATE"],
        
        # Authentication
        "password": ["VARCHAR(255)", "CHAR(60)"],
        "password_hash": ["VARCHAR(255)", "CHAR(60)"],
        "salt": ["VARCHAR(32)", "CHAR(32)"],
        "token": ["VARCHAR(255)", "TEXT"],
        "api_key": ["VARCHAR(100)", "VARCHAR(255)"],
        "refresh_token": ["VARCHAR(255)", "TEXT"],
        
        # Status and State
        "is_active": ["BOOLEAN", "TINYINT(1)"],
        "is_enabled": ["BOOLEAN", "TINYINT(1)"],
        "is_deleted": ["BOOLEAN", "TINYINT(1)"],
        "is_verified": ["BOOLEAN", "TINYINT(1)"],
        "is_default": ["BOOLEAN", "TINYINT(1)"],
        "is_public": ["BOOLEAN", "TINYINT(1)"],
        "is_featured": ["BOOLEAN", "TINYINT(1)"],
        
        # Numerical Values
        "amount": ["DECIMAL(10,2)", "NUMERIC(10,2)"],
        "quantity": ["INTEGER", "SMALLINT"],
        "total": ["DECIMAL(10,2)", "NUMERIC(10,2)"],
        "balance": ["DECIMAL(10,2)", "NUMERIC(10,2)"],
        "price": ["DECIMAL(10,2)", "NUMERIC(10,2)"],
        "cost": ["DECIMAL(10,2)", "NUMERIC(10,2)"],
        "rate": ["DECIMAL(5,2)", "NUMERIC(5,2)"],
        "percentage": ["DECIMAL(5,2)", "NUMERIC(5,2)"],
        "score": ["INTEGER", "DECIMAL(5,2)"],
        "rank": ["INTEGER", "SMALLINT"],
        "count": ["INTEGER", "BIGINT"],
        
        # Dimensions
        "size": ["VARCHAR(20)", "INTEGER"],
        "width": ["INTEGER", "DECIMAL(10,2)"],
        "height": ["INTEGER", "DECIMAL(10,2)"],
        "depth": ["INTEGER", "DECIMAL(10,2)"],
        "weight": ["DECIMAL(10,2)", "NUMERIC(10,2)"],
        
        # Financial
        "currency": ["CHAR(3)", "VARCHAR(3)"],
        "currency_code": ["CHAR(3)", "VARCHAR(3)"],
        "exchange_rate": ["DECIMAL(10,6)", "NUMERIC(10,6)"],
        "unit_price": ["DECIMAL(10,2)", "NUMERIC(10,2)"],
        "total_price": ["DECIMAL(10,2)", "NUMERIC(10,2)"],
        "tax_amount": ["DECIMAL(10,2)", "NUMERIC(10,2)"],
        
        # Media and Content
        "image": ["VARCHAR(255)", "TEXT"],
        "image_url": ["VARCHAR(255)", "TEXT"],
        "thumbnail": ["VARCHAR(255)", "TEXT"],
        "file_path": ["VARCHAR(255)", "TEXT"],
        "file_name": ["VARCHAR(255)"],
        "file_size": ["INTEGER", "BIGINT"],
        "file_type": ["VARCHAR(50)", "VARCHAR(100)"],
        "mime_type": ["VARCHAR(100)"],
        
        # Metadata
        "meta_title": ["VARCHAR(255)"],
        "meta_description": ["TEXT", "VARCHAR(500)"],
        "meta_keywords": ["TEXT", "VARCHAR(500)"],
        "tags": ["TEXT", "VARCHAR(500)", "JSONB"],
        "sequence": ["INTEGER", "SMALLINT"],
        "position": ["INTEGER", "SMALLINT"],
        "priority": ["INTEGER", "SMALLINT"],
        "version": ["INTEGER", "VARCHAR(50)"],
        
        # Configuration
        "settings": ["TEXT", "JSONB", "JSON"],
        "preferences": ["TEXT", "JSONB", "JSON"],
        "configuration": ["TEXT", "JSONB", "JSON"],
        "options": ["TEXT", "JSONB", "JSON"],
        "properties": ["TEXT", "JSONB", "JSON"],
        
        # System
        "ip_address": ["VARCHAR(45)", "INET"],
        "user_agent": ["TEXT", "VARCHAR(500)"],
        "browser": ["VARCHAR(100)"],
        "platform": ["VARCHAR(50)"],
        "device_type": ["VARCHAR(50)"],
        "language": ["CHAR(2)", "VARCHAR(5)"],
        "locale": ["VARCHAR(10)", "CHAR(5)"],
        "timezone": ["VARCHAR(50)"],
        
        # Statistics
        "views": ["INTEGER", "BIGINT"],
        "clicks": ["INTEGER", "BIGINT"],
        "impressions": ["INTEGER", "BIGINT"],
        "downloads": ["INTEGER", "BIGINT"],
        "rating": ["DECIMAL(3,2)", "NUMERIC(3,2)"],
        "votes": ["INTEGER", "BIGINT"],
        
        # Marketing
        "source": ["VARCHAR(100)"],
        "medium": ["VARCHAR(100)"],
        "campaign": ["VARCHAR(100)"],
        "referrer": ["VARCHAR(255)", "TEXT"],
        "utm_source": ["VARCHAR(100)"],
        "utm_medium": ["VARCHAR(100)"],
        "utm_campaign": ["VARCHAR(100)"],
        
        # Binary Data
        "data": ["BLOB", "BYTEA", "BINARY"],
        "content_blob": ["BLOB", "BYTEA", "BINARY"],
        "signature": ["BLOB", "BYTEA", "BINARY"],
        
        # Geographical
        "coordinates": ["POINT", "GEOMETRY"],
        "location": ["POINT", "GEOMETRY"],
        "area": ["POLYGON", "GEOMETRY"],
        "region": ["VARCHAR(100)", "GEOMETRY"],
        
        # Miscellaneous
        "color": ["VARCHAR(20)", "CHAR(7)"],
        "format": ["VARCHAR(50)"],
        "reason": ["VARCHAR(255)", "TEXT"],
        "result": ["VARCHAR(255)", "TEXT"],
        "response": ["TEXT", "JSONB"],
        "feedback": ["TEXT"],
        "hash": ["CHAR(32)", "CHAR(40)", "CHAR(64)"]
    }


def get_sql_column_names_and_types_chatgpt():
    column_types = {
        "id": ["INTEGER", "BIGINT", "SERIAL", "BIGSERIAL"],
        "uuid": ["UUID", "VARCHAR(36)", "CHAR(36)"],
        "name": ["VARCHAR(255)", "TEXT"],
        "first_name": ["VARCHAR(100)", "TEXT"],
        "last_name": ["VARCHAR(100)", "TEXT"],
        "full_name": ["VARCHAR(200)", "TEXT"],
        "username": ["VARCHAR(100)", "TEXT"],
        "password": ["VARCHAR(255)", "TEXT"],
        "email": ["VARCHAR(255)", "TEXT"],
        "phone": ["VARCHAR(20)", "TEXT"],
        "mobile": ["VARCHAR(20)", "TEXT"],
        "address": ["VARCHAR(255)", "TEXT"],
        "city": ["VARCHAR(100)", "TEXT"],
        "state": ["VARCHAR(100)", "TEXT"],
        "province": ["VARCHAR(100)", "TEXT"],
        "country": ["VARCHAR(100)", "TEXT"],
        "zip_code": ["VARCHAR(20)", "TEXT"],
        "postal_code": ["VARCHAR(20)", "TEXT"],
        "date_of_birth": ["DATE"],
        "birth_date": ["DATE"],
        "age": ["INTEGER", "SMALLINT"],
        "gender": ["VARCHAR(10)", "CHAR(1)"],
        "created_at": ["TIMESTAMP", "DATETIME"],
        "updated_at": ["TIMESTAMP", "DATETIME"],
        "deleted_at": ["TIMESTAMP", "DATETIME"],
        "last_login": ["TIMESTAMP", "DATETIME"],
        "status": ["VARCHAR(50)", "TEXT", "INTEGER"],
        "title": ["VARCHAR(255)", "TEXT"],
        "description": ["TEXT"],
        "content": ["TEXT"],
        "summary": ["TEXT"],
        "body": ["TEXT"],
        "excerpt": ["TEXT"],
        "price": ["DECIMAL(10,2)", "FLOAT"],
        "cost": ["DECIMAL(10,2)", "FLOAT"],
        "amount": ["DECIMAL(10,2)", "FLOAT"],
        "quantity": ["INTEGER", "SMALLINT"],
        "total": ["DECIMAL(10,2)", "FLOAT"],
        "subtotal": ["DECIMAL(10,2)", "FLOAT"],
        "tax": ["DECIMAL(10,2)", "FLOAT"],
        "discount": ["DECIMAL(10,2)", "FLOAT"],
        "rating": ["DECIMAL(2,1)", "FLOAT", "INTEGER"],
        "score": ["INTEGER", "SMALLINT"],
        "points": ["INTEGER", "SMALLINT"],
        "level": ["INTEGER", "SMALLINT"],
        "priority": ["VARCHAR(20)", "TEXT", "INTEGER"],
        "type": ["VARCHAR(50)", "TEXT"],
        "category": ["VARCHAR(100)", "TEXT"],
        "tag": ["VARCHAR(50)", "TEXT"],
        "tags": ["VARCHAR(255)", "TEXT"],
        "note": ["TEXT"],
        "comment": ["TEXT"],
        "message": ["TEXT"],
        "url": ["VARCHAR(2083)", "TEXT"],
        "link": ["VARCHAR(2083)", "TEXT"],
        "slug": ["VARCHAR(255)", "TEXT"],
        "ip_address": ["VARCHAR(45)", "TEXT"],
        "browser": ["VARCHAR(255)", "TEXT"],
        "device": ["VARCHAR(255)", "TEXT"],
        "operating_system": ["VARCHAR(255)", "TEXT"],
        "latitude": ["DECIMAL(9,6)", "FLOAT"],
        "longitude": ["DECIMAL(9,6)", "FLOAT"],
        "altitude": ["DECIMAL(9,6)", "FLOAT"],
        "speed": ["DECIMAL(9,6)", "FLOAT"],
        "direction": ["DECIMAL(9,6)", "FLOAT"],
        "timestamp": ["TIMESTAMP", "DATETIME"],
        "date": ["DATE"],
        "time": ["TIME"],
        "start_date": ["DATE"],
        "end_date": ["DATE"],
        "start_time": ["TIME"],
        "end_time": ["TIME"],
        "duration": ["INTEGER", "SMALLINT"],
        "expires_at": ["TIMESTAMP", "DATETIME"],
        "is_active": ["BOOLEAN", "TINYINT(1)"],
        "is_deleted": ["BOOLEAN", "TINYINT(1)"],
        "is_verified": ["BOOLEAN", "TINYINT(1)"],
        "is_published": ["BOOLEAN", "TINYINT(1)"],
        "is_enabled": ["BOOLEAN", "TINYINT(1)"],
        "is_admin": ["BOOLEAN", "TINYINT(1)"],
        "role": ["VARCHAR(50)", "TEXT"],
        "permissions": ["TEXT"],
        "settings": ["JSON", "TEXT"],
        "options": ["JSON", "TEXT"],
        "data": ["JSON", "TEXT"],
        "metadata": ["JSON", "TEXT"],
        "config": ["JSON", "TEXT"],
        "token": ["VARCHAR(255)", "TEXT"],
        "code": ["VARCHAR(50)", "TEXT"],
        "hash": ["VARCHAR(64)", "CHAR(64)"],
        "file_path": ["VARCHAR(255)", "TEXT"],
        "file_name": ["VARCHAR(255)", "TEXT"],
        "mime_type": ["VARCHAR(100)", "TEXT"],
        "size": ["BIGINT", "INTEGER"],
        "width": ["INTEGER", "SMALLINT"],
        "height": ["INTEGER", "SMALLINT"],
        "color": ["VARCHAR(20)", "TEXT"],
        "notes": ["TEXT"],
        "comments": ["TEXT"],
        "phone_number": ["VARCHAR(20)", "TEXT"],
        "mobile_number": ["VARCHAR(20)", "TEXT"],
        "fax_number": ["VARCHAR(20)", "TEXT"],
        "company": ["VARCHAR(255)", "TEXT"],
        "organization": ["VARCHAR(255)", "TEXT"],
        "department": ["VARCHAR(255)", "TEXT"],
        "position": ["VARCHAR(100)", "TEXT"],
        "job_title": ["VARCHAR(100)", "TEXT"],
        "salary": ["DECIMAL(10,2)", "FLOAT"],
        "date_hired": ["DATE"],
        "date_fired": ["DATE"],
        "supervisor_id": ["INTEGER", "BIGINT"],
        "manager_id": ["INTEGER", "BIGINT"],
        "customer_id": ["INTEGER", "BIGINT"],
        "order_id": ["INTEGER", "BIGINT"],
        "product_id": ["INTEGER", "BIGINT"],
        "category_id": ["INTEGER", "BIGINT"],
        "parent_id": ["INTEGER", "BIGINT"],
        "reference_id": ["INTEGER", "BIGINT"],
        "external_id": ["VARCHAR(100)", "TEXT"],
        "session_id": ["VARCHAR(255)", "TEXT"],
        "api_key": ["VARCHAR(255)", "TEXT"],
        "api_secret": ["VARCHAR(255)", "TEXT"],
        "access_token": ["VARCHAR(255)", "TEXT"],
        "refresh_token": ["VARCHAR(255)", "TEXT"],
        "user_agent": ["VARCHAR(255)", "TEXT"],
        "referrer": ["VARCHAR(2083)", "TEXT"],
        "locale": ["VARCHAR(10)", "TEXT"],
        "timezone": ["VARCHAR(50)", "TEXT"],
        "currency": ["VARCHAR(3)", "CHAR(3)"],
        "language": ["VARCHAR(10)", "TEXT"],
        "subject": ["VARCHAR(255)", "TEXT"],
        "attachment": ["VARCHAR(255)", "TEXT"],
        "signature": ["VARCHAR(255)", "TEXT"],
        "serial_number": ["VARCHAR(100)", "TEXT"],
        "model": ["VARCHAR(100)", "TEXT"],
        "version": ["VARCHAR(50)", "TEXT"],
        "manufacturer": ["VARCHAR(255)", "TEXT"],
        "brand": ["VARCHAR(100)", "TEXT"],
        "sku": ["VARCHAR(100)", "TEXT"],
        "upc": ["VARCHAR(12)", "TEXT"],
        "ean": ["VARCHAR(13)", "TEXT"],
        "isbn": ["VARCHAR(13)", "TEXT"],
        "release_date": ["DATE"],
        "publish_date": ["DATE"],
        "start_datetime": ["DATETIME", "TIMESTAMP"],
        "end_datetime": ["DATETIME", "TIMESTAMP"],
        "approved_at": ["TIMESTAMP", "DATETIME"],
        "published_at": ["TIMESTAMP", "DATETIME"],
        "archived_at": ["TIMESTAMP", "DATETIME"],
        "ip": ["VARCHAR(45)", "TEXT"],
        "mac_address": ["VARCHAR(17)", "CHAR(17)"],
        "accuracy": ["DECIMAL(5,2)", "FLOAT"],
        "heading": ["DECIMAL(5,2)", "FLOAT"],
        "nationality": ["VARCHAR(100)", "TEXT"],
        "marital_status": ["VARCHAR(20)", "TEXT"],
        "spouse_name": ["VARCHAR(255)", "TEXT"],
        "children": ["INTEGER", "SMALLINT"],
        "emergency_contact": ["VARCHAR(255)", "TEXT"],
        "relation": ["VARCHAR(50)", "TEXT"],
        "education": ["VARCHAR(255)", "TEXT"],
        "degree": ["VARCHAR(100)", "TEXT"],
        "major": ["VARCHAR(100)", "TEXT"],
        "gpa": ["DECIMAL(3,2)", "FLOAT"],
        "school": ["VARCHAR(255)", "TEXT"],
        "university": ["VARCHAR(255)", "TEXT"],
        "year_graduated": ["YEAR", "INTEGER"],
        "skills": ["TEXT"],
        "experience": ["TEXT"],
        "certification": ["VARCHAR(255)", "TEXT"],
        "license": ["VARCHAR(255)", "TEXT"],
        "availability": ["VARCHAR(50)", "TEXT"],
        "reference": ["TEXT"],
        "feedback": ["TEXT"],
        "progress": ["DECIMAL(5,2)", "FLOAT"],
        "milestone": ["VARCHAR(255)", "TEXT"],
        "due_date": ["DATE"],
        "completed_at": ["TIMESTAMP", "DATETIME"],
        "grade": ["VARCHAR(2)", "CHAR(2)"],
        "result": ["VARCHAR(50)", "TEXT"],
        "passed": ["BOOLEAN", "TINYINT(1)"],
        "failed": ["BOOLEAN", "TINYINT(1)"],
        "attempts": ["INTEGER", "SMALLINT"],
        "verified_at": ["TIMESTAMP", "DATETIME"],
        "reset_at": ["TIMESTAMP", "DATETIME"],
        "provider": ["VARCHAR(50)", "TEXT"],
        "platform": ["VARCHAR(50)", "TEXT"],
        "device_type": ["VARCHAR(50)", "TEXT"],
        "os_version": ["VARCHAR(50)", "TEXT"],
        "app_version": ["VARCHAR(50)", "TEXT"],
        "build_number": ["INTEGER", "SMALLINT"],
        "push_token": ["VARCHAR(255)", "TEXT"],
        "is_read": ["BOOLEAN", "TINYINT(1)"],
        "read_at": ["TIMESTAMP", "DATETIME"],
        "unread_count": ["INTEGER", "SMALLINT"],
        "last_message": ["TEXT"],
        "last_message_at": ["TIMESTAMP", "DATETIME"],
        "member_count": ["INTEGER", "SMALLINT"],
        "owner_id": ["INTEGER", "BIGINT"],
        "admin_id": ["INTEGER", "BIGINT"],
        "group_id": ["INTEGER", "BIGINT"],
        "team_id": ["INTEGER", "BIGINT"],
        "project_id": ["INTEGER", "BIGINT"],
        "task_id": ["INTEGER", "BIGINT"],
        "parent_task_id": ["INTEGER", "BIGINT"],
        "label": ["VARCHAR(50)", "TEXT"],
        "estimated_time": ["INTEGER", "SMALLINT"],
        "actual_time": ["INTEGER", "SMALLINT"],
        "file_id": ["INTEGER", "BIGINT"],
        "media_id": ["INTEGER", "BIGINT"],
        "image_id": ["INTEGER", "BIGINT"],
        "video_id": ["INTEGER", "BIGINT"],
        "document_id": ["INTEGER", "BIGINT"],
        "album_id": ["INTEGER", "BIGINT"],
        "playlist_id": ["INTEGER", "BIGINT"],
        "genre_id": ["INTEGER", "BIGINT"],
        "like_id": ["INTEGER", "BIGINT"],
        "share_id": ["INTEGER", "BIGINT"],
        "reaction": ["VARCHAR(20)", "TEXT"],
        "message_id": ["INTEGER", "BIGINT"],
        "chat_id": ["INTEGER", "BIGINT"],
        "thread_id": ["INTEGER", "BIGINT"],
        "post_id": ["INTEGER", "BIGINT"],
        "blog_id": ["INTEGER", "BIGINT"],
        "forum_id": ["INTEGER", "BIGINT"],
        "topic_id": ["INTEGER", "BIGINT"],
        "vote_id": ["INTEGER", "BIGINT"],
        "poll_id": ["INTEGER", "BIGINT"],
        "survey_id": ["INTEGER", "BIGINT"],
        "response_id": ["INTEGER", "BIGINT"],
        "question": ["TEXT"],
        "answer": ["TEXT"],
        "option": ["TEXT"],
        "choice": ["TEXT"],
        "selected": ["BOOLEAN", "TINYINT(1)"],
        "correct": ["BOOLEAN", "TINYINT(1)"],
        "max_score": ["INTEGER", "SMALLINT"],
        "min_score": ["INTEGER", "SMALLINT"],
        "average_score": ["DECIMAL(5,2)", "FLOAT"],
        "pass_score": ["DECIMAL(5,2)", "FLOAT"],
        "time_limit": ["INTEGER", "SMALLINT"],
        "time_taken": ["INTEGER", "SMALLINT"],
        "started_at": ["TIMESTAMP", "DATETIME"],
        "finished_at": ["TIMESTAMP", "DATETIME"],
    }
    return column_types

