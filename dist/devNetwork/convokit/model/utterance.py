from typing import Dict, Optional
from convokit.util import deprecation, warn
from .corpusComponent import CorpusComponent
from .speaker import Speaker


class Utterance(CorpusComponent):
    """Represents a single utterance in the dataset.

    :param id: the unique id of the utterance.
    :param speaker: the speaker giving the utterance.
    :param conversation_id: the id of the root utterance of the conversation.
    :param reply_to: id of the utterance this was a reply to.
    :param timestamp: timestamp of the utterance. Can be any
        comparable type.
    :param text: text of the utterance.

    :ivar id: the unique id of the utterance.
    :ivar speaker: the speaker giving the utterance.
    :ivar conversation_id: the id of the root utterance of the conversation.
    :ivar reply_to: id of the utterance this was a reply to.
    :ivar timestamp: timestamp of the utterance.
    :ivar text: text of the utterance.
    :ivar meta: A dictionary-like view object providing read-write access to
        utterance-level metadata.
    """

    def __init__(self, owner=None, id: Optional[str] = None, speaker: Optional[Speaker] = None,
                 user: Optional[Speaker] = None, conversation_id: Optional[str] = None,
                 root: Optional[str] = None, reply_to: Optional[str] = None,
                 timestamp: Optional[int] = None, text: str = '',
                 meta: Optional[Dict] = None):
        super().__init__(obj_type="utterance", owner=owner, id=id, meta=meta)
        speaker_ = speaker if speaker is not None else user
        self.speaker = speaker_
        if self.speaker is None:
            raise ValueError("No Speaker found: Utterance must be initialized with a Speaker.")
        self.user = speaker # for backwards compatbility
        self.conversation_id = conversation_id if conversation_id is not None else root
        if self.conversation_id is not None and not isinstance(self.conversation_id, str):
            warn("Utterance conversation_id must be a string: conversation_id of utterance with ID: {} "
                 "has been casted to a string.".format(self.id))
            self.conversation_id = str(self.conversation_id)
        self._root = self.conversation_id
        self.reply_to = reply_to
        self.timestamp = timestamp # int(timestamp) if timestamp is not None else timestamp
        if not isinstance(text, str):
            warn("Utterance text must be a string: text of utterance with ID: {} "
                 "has been casted to a string.".format(self.id))
            text = '' if text is None else str(text)
        self.text = text

    def _get_root(self):
        deprecation("utterance.root", "utterance.conversation_id")
        return self.conversation_id

    def _set_root(self, value: str):
        deprecation("utterance.root", "utterance.conversation_id")
        self.conversation_id = value
        # self._update_uid()

    root = property(_get_root, _set_root)

    def get_conversation(self):
        """
        Get the Conversation (identified by Utterance.conversation_id) this Utterance belongs to

        :return: a Conversation object
        """
        return self.owner.get_conversation(self.conversation_id)

    def get_speaker(self):
        """
        Get the Speaker that made this Utterance.

        :return: a Speaker object
        """

        return self.speaker

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, Utterance):
            return False
        try:
            return self.id == other.id and self.conversation_id == other.conversation_id and self.reply_to == other.reply_to and \
                   self.speaker == other.speaker and self.timestamp == other.timestamp and self.text == other.text
        except AttributeError: # for backwards compatibility with wikiconv
            return self.__dict__ == other.__dict__

    def __str__(self):
        return "Utterance(id: {}, conversation_id: {}, reply-to: {}, " \
               "speaker: {}, timestamp: {}, text: {}, vectors: {}, meta: {})".format(repr(self.id),
                                                                                    self.conversation_id,
                                                                                    self.reply_to,
                                                                                    self.speaker,
                                                                                    self.timestamp,
                                                                                    repr(self.text),
                                                                                    self.vectors,
                                                                                    self.meta)


