"""Unit tests for backend.database"""

import backend.database as db_mod


def test_save_and_retrieve_chat_message(temp_project_dirs):
    session_id = db_mod.new_session_id()
    msg_id = db_mod.save_chat_message(
        session_id=session_id,
        question="What is hostel fee?",
        answer="60000 per year.",
        confidence=0.9,
        source="fee_structure.pdf",
        intent="hostel",
        answered_by="rag",
    )
    assert isinstance(msg_id, int)

    history = db_mod.get_chat_history(session_id)
    assert len(history) == 1
    assert history[0].question == "What is hostel fee?"
    assert history[0].confidence == 0.9


def test_feedback_recorded_against_message(temp_project_dirs):
    session_id = db_mod.new_session_id()
    msg_id = db_mod.save_chat_message(session_id, "Q?", "A.", 0.8, None, "general", "rag")
    db_mod.record_feedback(msg_id, is_helpful=True, comment="Great answer")

    summary = db_mod.get_analytics_summary()
    assert summary["helpful_feedback"] == 1
    assert summary["not_helpful_feedback"] == 0


def test_analytics_summary_counts_sessions_and_messages(temp_project_dirs):
    s1, s2 = db_mod.new_session_id(), db_mod.new_session_id()
    db_mod.save_chat_message(s1, "Q1", "A1", 0.5, None, "general", "rag")
    db_mod.save_chat_message(s1, "Q2", "A2", 0.6, None, "general", "rag")
    db_mod.save_chat_message(s2, "Q3", "A3", 0.7, None, "general", "rag")

    summary = db_mod.get_analytics_summary()
    assert summary["total_messages"] == 3
    assert summary["total_sessions"] == 2


def test_faq_upsert_is_idempotent(temp_project_dirs):
    db_mod.upsert_faq("What is library timing?", "8 AM - 8 PM", "library")
    db_mod.upsert_faq("What is library timing?", "9 AM - 9 PM", "library")  # update

    faqs = db_mod.get_all_faqs()
    assert len(faqs) == 1
    assert faqs[0].answer == "9 AM - 9 PM"


def test_document_registration_and_listing(temp_project_dirs):
    db_mod.register_document("fee_structure.pdf", indexed=False)
    db_mod.register_document("fee_structure.pdf", indexed=True)  # update, not duplicate

    docs = db_mod.list_documents()
    assert len(docs) == 1
    assert docs[0].indexed is True

    db_mod.delete_document_record("fee_structure.pdf")
    assert db_mod.list_documents() == []


def test_common_questions_orders_by_frequency(temp_project_dirs):
    s = db_mod.new_session_id()
    for _ in range(3):
        db_mod.save_chat_message(s, "What is hostel fee?", "A", 0.9, None, "hostel", "faq")
    db_mod.save_chat_message(s, "What is library timing?", "A", 0.9, None, "library", "faq")

    common = db_mod.get_common_questions(limit=5)
    assert common[0][0] == "What is hostel fee?"
    assert common[0][1] == 3
