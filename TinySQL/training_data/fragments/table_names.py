from typing import List
from functools import lru_cache
import random
from .models import TableName

class TableInfo:
    def __init__(self, name: str, synonyms: List[str]):
        self.name = name
        # Replace underscores with spaces for each synonym for more natural English
        self.synonyms = [syn.replace("_", " ") for syn in synonyms]


@lru_cache(maxsize=1)
def get_TableInfo() -> List[TableInfo]:
    """Returns a list of common SQL table names with their synonyms in more natural English."""
    return [
        # Basic Data Tables
        TableInfo("data", ["information", "details"]),
        TableInfo("numbers", ["figures", "stats"]),
        TableInfo("text", ["words", "content"]),
        TableInfo("dates", ["calendar_entries", "time_points"]),
        TableInfo("times", ["moments", "hours"]),
        
        # Simple User Tables
        TableInfo("logins", ["sign-ins", "access_attempts"]),
        TableInfo("signup", ["registration", "joining"]),
        TableInfo("profile", ["personal_info", "bio"]),
        TableInfo("contact", ["reach_out", "get_in_touch"]),
        TableInfo("status", ["condition", "current_state"]),
        
        # Basic Content Tables
        TableInfo("images", ["pictures", "photos"]),
        TableInfo("videos", ["clips", "recordings"]),
        TableInfo("audio", ["sound_files", "voice_tracks"]),
        TableInfo("text_files", ["documents", "written_materials"]),
        TableInfo("links", ["references", "pointers"]),
        
        # Simple List Tables
        TableInfo("todo", ["tasks", "to-do_items"]),
        TableInfo("notes", ["reminders", "jottings"]),
        TableInfo("lists", ["groupings", "collections"]),
        TableInfo("items", ["things", "elements"]),
        TableInfo("favorites", ["bookmarks", "preferred_items"]),
        
        # Basic Location Tables
        TableInfo("place", ["spot", "location"]),
        TableInfo("address", ["residential_info", "delivery_point"]),
        TableInfo("map", ["layout", "guide"]),
        TableInfo("route", ["pathway", "directions"]),
        TableInfo("area", ["region", "zone"]),
        
        # Simple Time Tables
        TableInfo("schedule", ["plan", "timetable"]),
        TableInfo("day", ["date", "24_hours"]),
        TableInfo("week", ["7_day_period", "weekly_span"]),
        TableInfo("month", ["monthly_cycle", "30_day_period"]),
        TableInfo("year", ["annual_cycle", "12_month_period"]),
        
        # Basic Status Tables
        TableInfo("active", ["in_progress", "currently_running"]),
        TableInfo("pending", ["waiting", "on_hold"]),
        TableInfo("done", ["finished", "completed"]),
        TableInfo("hold", ["paused", "delayed"]),
        TableInfo("cancel", ["called_off", "terminated"]),
        
        # Simple Tracking Tables
        TableInfo("log", ["record", "chronicle"]),
        TableInfo("changes", ["revisions", "updates"]),
        TableInfo("backup", ["safekeeping_copy", "spare_version"]),
        TableInfo("archive", ["historical_storage", "record_depository"]),
        TableInfo("temp", ["short_term", "interim"]),
        
        # Basic Reference Tables
        TableInfo("types", ["categories", "kinds"]),
        TableInfo("codes", ["identifiers", "labels"]),
        TableInfo("names", ["titles", "designations"]),
        TableInfo("keys", ["unique_codes", "primary_references"]),
        TableInfo("values", ["quantities", "amounts"]),
        
        # Simple Join Tables
        TableInfo("links", ["connections", "relationships"]),
        TableInfo("maps", ["mappings", "cross_references"]),
        TableInfo("joins", ["combinations", "bridges"]),
        TableInfo("pairs", ["duos", "matched_sets"]),
        TableInfo("groups", ["clusters", "collectives"]),

        # User-related tables
        TableInfo("users", ["people", "members"]),
        TableInfo("user_profiles", ["member_details", "account_info"]),
        TableInfo("user_preferences", ["personal_settings", "user_options"]),
        TableInfo("user_settings", ["account_config", "profile_settings"]),
        TableInfo("user_roles", ["member_privileges", "user_positions"]),
        TableInfo("user_permissions", ["access_rights", "allowed_actions"]),
        TableInfo("user_sessions", ["login_periods", "active_connections"]),
        TableInfo("user_logs", ["activity_records", "usage_history"]),
        TableInfo("user_activity", ["actions_taken", "user_behaviors"]),
        TableInfo("user_metrics", ["user_statistics", "activity_data"]),
        
        # Authentication & Authorization
        TableInfo("accounts", ["user_accounts", "registered_profiles"]),
        TableInfo("roles", ["permission_levels", "user_groups"]),
        TableInfo("permissions", ["access_controls", "granted_rights"]),
        TableInfo("auth_tokens", ["security_keys", "login_tokens"]),
        TableInfo("login_attempts", ["access_tries", "sign_in_attempts"]),
        TableInfo("password_reset_tokens", ["recovery_keys", "reset_codes"]),
        TableInfo("access_logs", ["entry_history", "security_records"]),
        TableInfo("security_events", ["protection_incidents", "security_alerts"]),
        
        # Content Management
        TableInfo("posts", ["articles", "entries"]),
        TableInfo("articles", ["blog_posts", "writeups"]),
        TableInfo("comments", ["responses", "replies"]),
        TableInfo("categories", ["classifications", "groupings"]),
        TableInfo("tags", ["labels", "markers"]),
        TableInfo("media", ["multimedia", "digital_assets"]),
        TableInfo("attachments", ["additional_files", "linked_documents"]),
        TableInfo("documents", ["files", "written_records"]),
        TableInfo("pages", ["web_pages", "online_sections"]),
        TableInfo("content_revisions", ["version_history", "edit_records"]),
        
        # E-commerce
        TableInfo("products", ["goods", "offerings"]),
        TableInfo("product_categories", ["product_types", "merchandise_groups"]),
        TableInfo("product_variants", ["product_options", "item_variations"]),
        TableInfo("inventory", ["stock_levels", "available_items"]),
        TableInfo("orders", ["purchases", "transactions"]),
        TableInfo("order_items", ["purchased_products", "transaction_details"]),
        TableInfo("order_status", ["purchase_state", "progress_stage"]),
        TableInfo("shopping_cart", ["basket", "cart"]),
        TableInfo("cart_items", ["basket_contents", "cart_contents"]),
        TableInfo("wishlist", ["preferred_items", "saved_for_later"]),
        TableInfo("prices", ["cost_amounts", "rates"]),
        TableInfo("discounts", ["price_reductions", "special_offers"]),
        TableInfo("promotions", ["campaign_offers", "marketing_deals"]),
        TableInfo("coupons", ["discount_codes", "vouchers"]),
        
        # Customer Relations
        TableInfo("customers", ["clients", "buyers"]),
        TableInfo("customer_addresses", ["delivery_locations", "client_addresses"]),
        TableInfo("customer_preferences", ["client_choices", "patron_settings"]),
        TableInfo("customer_support_tickets", ["help_requests", "service_cases"]),
        TableInfo("feedback", ["opinions", "user_input"]),
        TableInfo("complaints", ["grievances", "customer_issues"]),
        TableInfo("reviews", ["critiques", "ratings"]),
        TableInfo("ratings", ["scores", "evaluations"]),
        
        # Financial
        TableInfo("transactions", ["financial_events", "monetary_records"]),
        TableInfo("payments", ["settlements", "fund_transfers"]),
        TableInfo("payment_methods", ["ways_to_pay", "payment_options"]),
        TableInfo("invoices", ["bills", "statements"]),
        TableInfo("invoice_items", ["billing_details", "charge_items"]),
        TableInfo("refunds", ["money_returns", "reimbursements"]),
        TableInfo("subscriptions", ["recurring_services", "ongoing_plans"]),
        TableInfo("subscription_plans", ["membership_packages", "plan_options"]),
        TableInfo("billing_cycles", ["payment_intervals", "invoicing_periods"]),
        TableInfo("payment_history", ["transaction_log", "past_payments"]),
        
        # Shipping
        TableInfo("shipping_addresses", ["mailing_points", "delivery_locations"]),
        TableInfo("shipping_methods", ["delivery_options", "shipment_types"]),
        TableInfo("shipping_zones", ["service_areas", "coverage_regions"]),
        TableInfo("shipping_rates", ["delivery_costs", "freight_charges"]),
        TableInfo("delivery_status", ["shipment_progress", "tracking_state"]),
        TableInfo("tracking_info", ["package_updates", "shipment_details"]),
        
        # Employee/HR
        TableInfo("employees", ["staff_members", "workforce"]),
        TableInfo("departments", ["branches", "sections"]),
        TableInfo("positions", ["roles", "job_titles"]),
        TableInfo("salary_info", ["pay_details", "compensation_data"]),
        TableInfo("attendance", ["presence_records", "time_logs"]),
        TableInfo("leave_requests", ["time_off_applications", "absence_forms"]),
        TableInfo("performance_reviews", ["staff_evaluations", "work_assessments"]),
        TableInfo("training_records", ["learning_logs", "development_history"]),
        
        # Communication
        TableInfo("messages", ["notes", "communications"]),
        TableInfo("conversations", ["discussions", "dialogues"]),
        TableInfo("chat_rooms", ["group_chats", "conversation_spaces"]),
        TableInfo("notifications", ["alerts", "updates"]),
        TableInfo("email_templates", ["message_formats", "mail_blueprints"]),
        TableInfo("sms_logs", ["text_records", "mobile_messages"]),
        TableInfo("push_notifications", ["app_alerts", "mobile_prompts"]),
        
        # Analytics
        TableInfo("page_views", ["site_visits", "view_counts"]),
        TableInfo("events", ["occurrences", "happenings"]),
        TableInfo("event_logs", ["activity_records", "incident_logs"]),
        TableInfo("metrics", ["measurements", "performance_indicators"]),
        TableInfo("analytics_data", ["usage_stats", "analysis_information"]),
        TableInfo("user_behavior", ["visitor_actions", "interaction_patterns"]),
        TableInfo("conversion_funnel", ["sales_path", "user_journey"]),
        TableInfo("ab_test_results", ["experiment_outcomes", "test_findings"]),
        
        # Location
        TableInfo("countries", ["nations", "states"]),
        TableInfo("regions", ["territories", "zones"]),
        TableInfo("cities", ["municipalities", "towns"]),
        TableInfo("addresses", ["locations", "places"]),
        TableInfo("locations", ["coordinates", "spots"]),
        TableInfo("geo_data", ["geographic_info", "location_details"]),
        TableInfo("zip_codes", ["postal_codes", "mail_areas"]),
        TableInfo("postal_codes", ["zip_codes", "mail_routes"]),
        
        # System
        TableInfo("settings", ["configurations", "preferences"]),
        TableInfo("configurations", ["system_options", "setup_details"]),
        TableInfo("system_logs", ["application_history", "operation_records"]),
        TableInfo("error_logs", ["failure_reports", "exception_records"]),
        TableInfo("audit_trail", ["change_log", "monitoring_history"]),
        TableInfo("cache", ["temp_storage", "speed_buffer"]),
        TableInfo("jobs", ["tasks", "processes"]),
        TableInfo("queues", ["task_lineups", "job_lists"]),
        TableInfo("scheduled_tasks", ["timed_jobs", "planned_operations"]),

        # Project Management
        TableInfo("projects", ["initiatives", "ventures"]),
        TableInfo("project_phases", ["stage_details", "phased_tasks"]),
        TableInfo("milestones", ["key_events", "project_markers"]),
        TableInfo("deliverables", ["project_outputs", "end_products"]),
        TableInfo("project_resources", ["assets", "support_materials"]),
        TableInfo("project_budgets", ["cost_plans", "fund_allocations"]),
        TableInfo("project_risks", ["potential_issues", "threat_assessments"]),
        TableInfo("project_stakeholders", ["interested_parties", "project_contacts"]),
        TableInfo("project_timeline", ["schedule", "work_plan"]),
        TableInfo("task_dependencies", ["task_links", "prerequisite_steps"]),

        # Marketing
        TableInfo("campaigns", ["promotional_efforts", "marketing_strategies"]),
        TableInfo("campaign_budgets", ["marketing_spend", "promotion_funds"]),
        TableInfo("campaign_metrics", ["marketing_stats", "success_measures"]),
        TableInfo("marketing_channels", ["promotion_outlets", "advertising_paths"]),
        TableInfo("marketing_assets", ["promo_materials", "brand_resources"]),
        TableInfo("audience_segments", ["target_groups", "consumer_sections"]),
        TableInfo("marketing_content", ["promotional_content", "campaign_materials"]),
        TableInfo("lead_sources", ["prospect_origins", "referral_paths"]),
        TableInfo("marketing_goals", ["campaign_objectives", "promo_targets"]),
        TableInfo("brand_assets", ["branding_materials", "visual_identity"]),
        
        # Social Media
        TableInfo("social_posts", ["platform_updates", "public_shares"]),
        TableInfo("social_engagement", ["interaction_metrics", "user_involvement"]),
        TableInfo("social_followers", ["audience_members", "platform_subscribers"]),
        TableInfo("social_mentions", ["brand_shoutouts", "named_references"]),
        TableInfo("social_campaigns", ["platform_promotions", "social_drives"]),
        TableInfo("social_analytics", ["platform_metrics", "engagement_data"]),
        TableInfo("social_schedules", ["posting_calendar", "release_timeline"]),
        TableInfo("hashtag_tracking", ["tag_monitoring", "keyword_watching"]),
        TableInfo("social_interactions", ["comments_likes", "audience_activity"]),
        TableInfo("social_influencers", ["content_creators", "brand_advocates"]),
        
        # Customer Service
        TableInfo("service_requests", ["support_tickets", "help_needs"]),
        TableInfo("service_agents", ["support_staff", "assist_team"]),
        TableInfo("service_queues", ["ticket_line", "helpdesk_pipeline"]),
        TableInfo("resolution_times", ["response_speeds", "handling_durations"]),
        TableInfo("service_levels", ["support_tiers", "assistance_plans"]),
        TableInfo("knowledge_base", ["help_articles", "support_docs"]),
        TableInfo("faq_entries", ["common_questions", "frequent_inquiries"]),
        TableInfo("support_channels", ["contact_methods", "helpdesk_routes"]),
        TableInfo("escalation_rules", ["priority_guidelines", "routing_conditions"]),
        TableInfo("customer_satisfaction", ["service_feedback", "support_ratings"]),
        
        # Product Development
        TableInfo("product_features", ["capabilities", "functionalities"]),
        TableInfo("feature_requests", ["enhancement_ideas", "improvement_suggestions"]),
        TableInfo("product_roadmap", ["development_timeline", "future_plans"]),
        TableInfo("product_versions", ["releases", "updates"]),
        TableInfo("product_specs", ["requirements", "technical_details"]),
        TableInfo("product_feedback", ["user_comments", "feature_reviews"]),
        TableInfo("product_bugs", ["defects", "known_issues"]),
        TableInfo("product_testing", ["quality_checks", "verification_steps"]),
        TableInfo("product_documentation", ["user_guides", "product_manuals"]),
        TableInfo("product_components", ["modules", "building_blocks"]),
        
        # Learning Management
        TableInfo("courses", ["learning_paths", "training_modules"]),
        TableInfo("course_modules", ["lessons", "training_sections"]),
        TableInfo("course_materials", ["study_documents", "learning_resources"]),
        TableInfo("assessments", ["tests", "evaluations"]),
        TableInfo("quiz_questions", ["test_items", "assessment_prompts"]),
        TableInfo("student_progress", ["learner_status", "training_advancement"]),
        TableInfo("certifications", ["credentials", "qualifications"]),
        TableInfo("learning_paths", ["education_tracks", "course_outlines"]),
        TableInfo("instructors", ["teachers", "facilitators"]),
        TableInfo("class_schedules", ["training_calendar", "session_timetable"]),
        
        # Time Management
        TableInfo("calendars", ["date_schedules", "timetables"]),
        TableInfo("appointments", ["meetings", "booked_slots"]),
        TableInfo("availability", ["free_time", "open_slots"]),
        TableInfo("time_blocks", ["scheduled_segments", "allocated_periods"]),
        TableInfo("recurring_events", ["regular_meetings", "periodic_activities"]),
        TableInfo("time_preferences", ["scheduling_options", "time_settings"]),
        TableInfo("calendar_sync", ["schedule_integration", "timetable_sharing"]),
        TableInfo("time_zones", ["regional_offsets", "location_hours"]),
        TableInfo("event_reminders", ["alerts", "notifications"]),
        TableInfo("event_categories", ["activity_types", "meeting_kinds"]),
        
        # Quality Management
        TableInfo("quality_metrics", ["performance_signs", "quality_indicators"]),
        TableInfo("quality_checks", ["inspections", "assurance_tests"]),
        TableInfo("quality_issues", ["product_defects", "problems"]),
        TableInfo("corrective_actions", ["fixes", "improvement_steps"]),
        TableInfo("quality_standards", ["requirements", "benchmarks"]),
        TableInfo("quality_documents", ["procedures", "guidelines"]),
        TableInfo("audit_results", ["inspection_outcomes", "review_findings"]),
        TableInfo("quality_teams", ["review_groups", "inspection_personnel"]),
        TableInfo("quality_training", ["skill_development", "competency_building"]),
        TableInfo("quality_reports", ["assessment_documents", "performance_summaries"]),
        
        # Document Management
        TableInfo("document_versions", ["file_editions", "revision_history"]),
        TableInfo("document_metadata", ["file_details", "document_info"]),
        TableInfo("document_permissions", ["access_levels", "sharing_options"]),
        TableInfo("document_categories", ["file_types", "content_groups"]),
        TableInfo("document_templates", ["format_blueprints", "layout_structures"]),
        TableInfo("document_workflows", ["approval_paths", "review_cycles"]),
        TableInfo("document_storage", ["file_repositories", "content_locations"]),
        TableInfo("document_sharing", ["file_exchange", "access_links"]),
        TableInfo("document_links", ["reference_paths", "related_files"]),
        TableInfo("document_history", ["change_log", "edit_records"]),
        
        # Risk Management
        TableInfo("risk_assessments", ["threat_evaluations", "vulnerability_checks"]),
        TableInfo("risk_mitigation", ["preventive_measures", "control_strategies"]),
        TableInfo("risk_incidents", ["security_events", "unwanted_occurrences"]),
        TableInfo("risk_categories", ["threat_types", "risk_classes"]),
        TableInfo("risk_impacts", ["consequence_outcomes", "effect_analysis"]),
        TableInfo("risk_probabilities", ["likelihood_estimates", "chance_levels"]),
        TableInfo("risk_controls", ["safeguards", "shielding_methods"]),
        TableInfo("risk_monitoring", ["threat_tracking", "ongoing_surveillance"]),
        TableInfo("risk_reports", ["incident_summaries", "analysis_documents"]),
        TableInfo("risk_owners", ["responsible_people", "assigned_parties"]),
        
        # API Management
        TableInfo("api_endpoints", ["service_urls", "integration_points"]),
        TableInfo("api_versions", ["release_levels", "service_editions"]),
        TableInfo("api_keys", ["access_tokens", "authentication_keys"]),
        TableInfo("api_usage", ["service_activity", "endpoint_calls"]),
        TableInfo("api_documentation", ["service_guides", "api_overview"]),
        TableInfo("api_permissions", ["access_rules", "usage_rights"]),
        TableInfo("api_rate_limits", ["usage_caps", "request_boundaries"]),
        TableInfo("api_logs", ["activity_logs", "api_history"]),
        TableInfo("api_performance", ["service_speed", "efficiency_metrics"]),
        TableInfo("api_errors", ["service_failures", "endpoint_issues"]),
        
        # Reporting
        TableInfo("report_templates", ["document_skeletons", "output_blueprints"]),
        TableInfo("report_schedules", ["generation_plans", "report_timing"]),
        TableInfo("report_parameters", ["input_criteria", "filter_options"]),
        TableInfo("report_outputs", ["generated_files", "presentation_results"]),
        TableInfo("report_distribution", ["delivery_lists", "output_recipients"]),
        TableInfo("report_access", ["view_permissions", "sharing_rules"]),
        TableInfo("report_history", ["generation_record", "previous_outputs"]),
        TableInfo("report_metrics", ["report_statistics", "performance_data"]),
        TableInfo("report_categories", ["document_types", "topic_labels"]),
        TableInfo("report_comments", ["feedback_notes", "remark_entries"]),
        
        # Workflow Management
        TableInfo("workflow_definitions", ["process_blueprints", "procedure_designs"]),
        TableInfo("workflow_steps", ["process_stages", "action_sequence"]),
        TableInfo("workflow_rules", ["process_guidelines", "logic_conditions"]),
        TableInfo("workflow_assignments", ["task_distribution", "role_allocations"]),
        TableInfo("workflow_status", ["current_step", "process_progress"]),
        TableInfo("workflow_history", ["execution_log", "past_runs"]),
        TableInfo("workflow_metrics", ["process_stats", "efficiency_data"]),
        TableInfo("workflow_templates", ["standard_processes", "procedure_patterns"]),
        TableInfo("workflow_triggers", ["start_conditions", "process_initiators"]),
        TableInfo("workflow_notifications", ["process_alerts", "task_updates"]),
        
        # Content Delivery
        TableInfo("content_nodes", ["distribution_points", "delivery_servers"]),
        TableInfo("content_routes", ["paths_to_users", "network_flows"]),
        TableInfo("content_caching", ["temporary_storage", "faster_access"]),
        TableInfo("content_security", ["protection_measures", "access_safeguards"]),
        TableInfo("content_optimization", ["performance_tuning", "speed_enhancements"]),
        TableInfo("content_metrics", ["delivery_stats", "monitoring_data"]),
        TableInfo("content_availability", ["uptime_monitoring", "access_tracking"]),
        TableInfo("content_backups", ["duplicate_storage", "backup_copies"]),
        TableInfo("content_restrictions", ["usage_limits", "access_constraints"]),
        TableInfo("content_scheduling", ["timed_delivery", "planned_distribution"]),
        
        # Search Management
        TableInfo("search_indexes", ["content_catalogs", "lookup_structures"]),
        TableInfo("search_queries", ["lookup_requests", "search_requests"]),
        TableInfo("search_results", ["found_items", "matched_records"]),
        TableInfo("search_filters", ["refinement_options", "filter_criteria"]),
        TableInfo("search_rankings", ["result_order", "priority_scores"]),
        TableInfo("search_suggestions", ["query_hints", "lookup_tips"]),
        TableInfo("search_synonyms", ["similar_terms", "alternate_words"]),
        TableInfo("search_history", ["previous_lookups", "query_log"]),
    ]


def get_sql_table_names_with_synonyms() -> List[TableInfo]:
    """Returns a list of common SQL table names with their synonyms in more natural English."""
    return get_TableInfo()


def get_sql_table_name() -> TableName:
    """Returns a random table name from the list."""
    all_tables = get_TableInfo()

    the_table = random.choice(all_tables)
    the_synonym = random.choice(the_table.synonyms)

    answer = TableName
    answer.name = the_table.name
    answer.synonym = the_synonym
    return answer


def get_sql_table_name_count() -> int:
    """Returns the total number of unique table names."""
    return len(get_TableInfo())