# =========================================
# INSTALL LIBRARIES
# =========================================

!pip install -q transformers accelerate gradio wikipedia sentencepiece

# =========================================
# IMPORTS
# =========================================

import wikipedia
import gradio as gr

from transformers import pipeline

# =========================================
# LOAD FREE AI MODEL
# =========================================

generator = pipeline(
    "text-generation",
    model="google/flan-t5-base"
)

# =========================================
# RESEARCH AGENT
# =========================================

def research_agent(topic):

    try:

        research_data = wikipedia.summary(
            topic,
            sentences=10
        )

        return research_data

    except Exception as e:

        return f"Research Error: {str(e)}"

# =========================================
# SUMMARY AGENT
# =========================================

def summary_agent(text):

    prompt = f"""
    Summarize this:

    {text}
    """

    result = generator(
        prompt,
        max_length=200
    )

    return result[0]["generated_text"]

# =========================================
# NOTES AGENT
# =========================================

def notes_agent(text):

    prompt = f"""
    Create detailed study notes:

    {text}
    """

    result = generator(
        prompt,
        max_length=150
    )

    return result[0]["generated_text"]

# =========================================
# QUIZ AGENT
# =========================================

def quiz_agent(text):

    prompt = f"""
    Generate 5 quiz questions from this:

    {text}
    """

    result = generator(
        prompt,
        max_length=175
    )

    return result[0]["generated_text"]

# =========================================
# INSIGHTS AGENT
# =========================================

def insights_agent(text):

    prompt = f"""
    Extract key insights:

    {text}
    """

    result = generator(
        prompt,
        max_length=170
    )

    return result[0]["generated_text"]

# =========================================
# MAIN WORKFLOW
# =========================================

def ai_research_assistant(topic):

    try:

        # RESEARCH
        research = research_agent(topic)

        # SUMMARY
        summary = summary_agent(research)

        # NOTES
        notes = notes_agent(research)

        # QUIZ
        quiz = quiz_agent(research)

        # INSIGHTS
        insights = insights_agent(research)

        return (
            research,
            summary,
            notes,
            quiz,
            insights
        )

    except Exception as e:

        error = str(e)

        return (
            error,
            error,
            error,
            error,
            error
        )

# =========================================
# GRADIO UI
# =========================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# Agentic AI Research Assistant")

    gr.Markdown("""
    ### Features
    ✅ Research Topics
    ✅ Generate Summaries
    ✅ Create Notes
    ✅ Generate Quiz Questions
    ✅ Extract Insights
    ✅ Fully FREE
    """)

    topic_input = gr.Textbox(
        label="Enter Research Topic",
        placeholder="Artificial Intelligence"
    )

    run_button = gr.Button("🚀 Start Research")

    with gr.Tab("📖 Research"):
        research_output = gr.Textbox(lines=12)

    with gr.Tab("📝 Summary"):
        summary_output = gr.Textbox(lines=10)

    with gr.Tab("📚 Notes"):
        notes_output = gr.Textbox(lines=15)

    with gr.Tab("❓ Quiz"):
        quiz_output = gr.Textbox(lines=12)

    with gr.Tab("💡 Insights"):
        insights_output = gr.Textbox(lines=10)

    run_button.click(
        fn=ai_research_assistant,
        inputs=topic_input,
        outputs=[
            research_output,
            summary_output,
            notes_output,
            quiz_output,
            insights_output
        ]
    )

# =========================================
# LAUNCH APP
# =========================================

demo.launch(share=True)
