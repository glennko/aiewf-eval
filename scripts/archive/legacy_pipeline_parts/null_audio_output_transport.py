#
# Null Audio Output Transport for Pipecat
#
# A minimal output transport that inherits BaseOutputTransport's speaking detection
# logic (BotStartedSpeakingFrame/BotStoppedSpeakingFrame) but discards audio output.
#
# This is useful for pipelines that need speaking detection without actual audio playback,
# such as evaluation/test pipelines or speech-to-speech model testing.
#

from pipecat.frames.frames import OutputAudioRawFrame, StartFrame
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams


class NullAudioOutputTransport(BaseOutputTransport):
    """Output transport that tracks audio for speaking detection but discards output.

    This transport extends BaseOutputTransport to inherit its MediaSender logic,
    which automatically generates BotStartedSpeakingFrame and BotStoppedSpeakingFrame
    based on audio output timing. However, it doesn't actually play or send the audio
    anywhere - it just discards it.

    This is useful for:
    - Test/evaluation pipelines where you don't need audio playback
    - Speech-to-speech model testing where you only need transcripts
    - Pipelines that need speaking state tracking without audio output hardware

    The key mechanism inherited from BaseOutputTransport:
    - MediaSender tracks TTSAudioRawFrame timing
    - After BOT_VAD_STOP_SECS (0.35s) of no audio, generates BotStoppedSpeakingFrame
    - BotStoppedSpeakingFrame flows upstream to trigger response finalization
    """

    def __init__(self, params: TransportParams, **kwargs):
        """Initialize the null audio output transport.

        Args:
            params: Transport configuration parameters. Should have audio_out_enabled=True
                    and appropriate sample rate settings.
            **kwargs: Additional arguments passed to BaseOutputTransport.
        """
        super().__init__(params, **kwargs)

    async def start(self, frame: StartFrame):
        """Start the transport and initialize the MediaSender.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        # Call set_transport_ready to initialize MediaSender which handles
        # BotStartedSpeakingFrame/BotStoppedSpeakingFrame generation
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Discard the audio frame but return True to continue tracking.

        The MediaSender in BaseOutputTransport uses the return value to decide
        whether to push frames downstream. We return True so the speaking
        detection logic continues to work properly.

        Args:
            frame: The audio frame to "write" (actually discarded).

        Returns:
            True always, indicating the frame was "successfully written".
        """
        # Don't actually play/send the audio - just discard it
        # Return True so MediaSender continues to track speaking state
        return True
