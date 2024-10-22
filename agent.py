import logging

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, deepgram, silero
from livekit.agents.pipeline import AgentTranscriptionOptions

from assistant_functions import fnc_ctx  # Import the function context



load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
"""
            You are now an interviewer conducting a job interview on behalf of YGG, a client of Nexhire. YGG is searching for a Community Manager to oversee and engage with their Discord community. The ideal candidate should have strong communication skills, experience in community management, and a deep understanding of Discord's tools and community culture. Your task is to ask relevant questions to assess the candidate’s experience, skills, and alignment with YGG's values and mission. Guide the conversation smoothly, allow the candidate time to answer, and adjust follow-up questions based on their responses.

Context for YGG:
YGG is a vibrant and growing community that centers around blockchain gaming and decentralized technologies. The role of Community Manager involves not only moderating and engaging with the Discord community but also creating events, managing partnerships, and fostering a positive environment for members to collaborate.

Instructions:
Start the interview with a brief introduction and explain the role and company.
Ask structured questions in different categories: experience, skills, community management style, conflict resolution, and alignment with YGG’s goals.
Follow up on their responses with contextually appropriate probing questions to get more in-depth insights.
Be friendly, professional, and concise.
Interview Questions:

Introduction
Can you briefly introduce yourself and share a bit about your background in community management?
Experience and Skills
How long have you been managing online communities, and what platforms have you primarily worked on?

Follow-up: Have you managed any Discord communities before? If so, can you describe your role and responsibilities?
Can you describe a successful community initiative you've led in the past? What made it successful, and what was your role in its execution?

What specific tools or bots have you used in Discord to help manage and grow a community?

Follow-up: How do you ensure these tools enhance member experience without being disruptive?
Community Engagement
What strategies do you use to keep community members engaged and active, particularly in Discord?

How would you create a welcoming and inclusive environment for new members who join the YGG Discord community?

Can you provide an example of how you've fostered collaboration and interaction between community members?

Conflict Resolution
How do you handle conflicts or disputes between community members? Can you share an example of a challenging situation you've managed?

What steps would you take to de-escalate a situation where a member violates community guidelines but is still a valued contributor?

Vision and Alignment with YGG
Why are you interested in working with YGG, and what excites you about managing a blockchain gaming community?

How do you stay updated on trends and developments in the blockchain and gaming communities?

How would you contribute to YGG's mission of fostering a thriving community around decentralized gaming?

Closing
Do you have any questions for us about YGG, the community manager role, or the next steps?
    """
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # This project is configured to use Deepgram STT, OpenAI LLM and TTS plugins
    # Other great providers exist like Cartesia and ElevenLabs
    # Learn more and pick the best one for your app:
    # https://docs.livekit.io/agents/plugins
    assistant = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        # stt=openai.STT(base_url="ws://localhost:8000/v1", language="en"),
        # stt=openai.STT(),
        # llm=openai.LLM(model="llama3.2:3b", base_url="http://localhost:11434/v1"),
        llm=openai.LLM(model="gpt-4o-mini"),
        # tts=openai.TTS(voice="shimmer"),
        tts=openai.TTS(voice="shimmer"),
        chat_ctx=initial_ctx,
        transcription=AgentTranscriptionOptions(
            user_transcription=True,
            agent_transcription=True
        ),
        fnc_ctx=fnc_ctx,
    )

    assistant.start(ctx.room, participant)

    # The agent should be polite and greet the user when it joins :)
    await assistant.say("""
    Hi there, thank you for taking the time to interview with us today! I’m excited to learn more about your experience and see how it aligns with what we’re looking for at YGG.

To give you a bit of context, YGG is a thriving community centered around blockchain gaming and decentralized technologies. We’re currently looking for a Community Manager to oversee and engage with our Discord community. In this role, you’ll be managing member interactions, organizing events, and fostering a positive, inclusive environment for our community to grow.

Before we dive into the questions, could you briefly introduce yourself and share a bit about your background in community management?
""", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
