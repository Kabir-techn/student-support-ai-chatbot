"""Unit tests for backend.chatbot (orchestration: FAQ -> RAG -> memory -> persistence)"""

from backend.chatbot import Chatbot, detect_intent


def test_detect_intent_matches_keywords():
    assert detect_intent("What is the hostel fee?") == "fees"
    assert detect_intent("Is hostel available for girls?") == "hostel"
    assert detect_intent("When is the semester exam?") == "examinations"
    assert detect_intent("Tell me a joke") == "general"


def test_chat_hits_faq_for_known_question(temp_project_dirs):
    bot = Chatbot()
    response = bot.chat("What is hostel fee?")
    assert response.answered_by == "faq"
    assert response.confidence > 0.9
    assert "60,000" in response.answer


def test_chat_persists_message_and_returns_message_id(temp_project_dirs):
    bot = Chatbot()
    response = bot.chat("What is library timing?")
    assert response.message_id is not None

    from backend.database import get_chat_history

    history = get_chat_history(response.session_id)
    assert len(history) == 1
    assert history[0].answer == response.answer


def test_chat_falls_back_when_out_of_domain(temp_project_dirs):
    bot = Chatbot()
    response = bot.chat("What is the airspeed velocity of an unladen swallow?")
    assert response.answered_by == "fallback"
    assert "not confident" in response.answer.lower()


def test_chat_reuses_session_id_across_turns(temp_project_dirs):
    bot = Chatbot()
    r1 = bot.chat("What is library timing?")
    r2 = bot.chat("Thanks!", session_id=r1.session_id)
    assert r1.session_id == r2.session_id

    from backend.memory import conversation_memory

    history_text = conversation_memory.get_history_text(r1.session_id)
    assert "library" in history_text.lower()
