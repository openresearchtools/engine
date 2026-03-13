use crate::audio_assembler::{
    assemble_turns_with_words, turns_to_markdown, AssembleOptions, SpeakerSpan, SpeakerTurn,
    TranscriptPiece,
};
use crate::llama_bridge::{
    AudioSessionEventOwned, LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_SPAN_COMMIT,
    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_PIECE_COMMIT,
    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_RESULT_JSON,
    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_WORD_COMMIT,
};

#[derive(Clone, Debug)]
pub struct OrchestratorSnapshot {
    pub spans: Vec<SpeakerSpan>,
    pub pieces: Vec<TranscriptPiece>,
    pub words: Vec<TranscriptPiece>,
    pub turns: Vec<SpeakerTurn>,
    pub markdown: String,
    pub latest_transcription_json: Option<String>,
}

pub struct DiarizedTranscriptOrchestrator {
    sample_rate_hz: u32,
    options: AssembleOptions,
    spans: Vec<SpeakerSpan>,
    pieces: Vec<TranscriptPiece>,
    words: Vec<TranscriptPiece>,
    turns: Vec<SpeakerTurn>,
    markdown: String,
    latest_transcription_json: Option<String>,
}

impl DiarizedTranscriptOrchestrator {
    pub fn new(sample_rate_hz: u32) -> Self {
        Self {
            sample_rate_hz,
            options: AssembleOptions::default(),
            spans: Vec::new(),
            pieces: Vec::new(),
            words: Vec::new(),
            turns: Vec::new(),
            markdown: String::new(),
            latest_transcription_json: None,
        }
    }

    pub fn with_options(sample_rate_hz: u32, options: AssembleOptions) -> Self {
        let mut out = Self::new(sample_rate_hz);
        out.options = options;
        out
    }

    pub fn ingest_event(&mut self, event: &AudioSessionEventOwned) -> bool {
        let mut changed = false;
        match event.kind {
            LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_SPAN_COMMIT => {
                let speaker = if !event.text.trim().is_empty() {
                    event.text.trim().to_string()
                } else if event.speaker_id >= 0 {
                    format!("SPEAKER_{:02}", event.speaker_id)
                } else {
                    "UNASSIGNED".to_string()
                };
                changed |= insert_unique_span(
                    &mut self.spans,
                    SpeakerSpan {
                        speaker,
                        start_sample: event.start_sample,
                        end_sample: event.end_sample,
                    },
                );
            }
            LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_PIECE_COMMIT => {
                changed |= insert_unique_piece(
                    &mut self.pieces,
                    TranscriptPiece {
                        start_sample: event.start_sample,
                        end_sample: event.end_sample,
                        text: event.text.clone(),
                    },
                );
            }
            LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_WORD_COMMIT => {
                changed |= insert_unique_piece(
                    &mut self.words,
                    TranscriptPiece {
                        start_sample: event.start_sample,
                        end_sample: event.end_sample,
                        text: event.text.clone(),
                    },
                );
            }
            LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_RESULT_JSON => {
                self.latest_transcription_json = Some(event.text.clone());
            }
            _ => {}
        }

        if changed && !self.spans.is_empty() && !self.pieces.is_empty() {
            self.turns =
                assemble_turns_with_words(&self.spans, &self.pieces, &self.words, &self.options);
            self.markdown = turns_to_markdown(&self.turns, self.sample_rate_hz);
        }

        changed
    }

    pub fn snapshot(&self) -> OrchestratorSnapshot {
        OrchestratorSnapshot {
            spans: self.spans.clone(),
            pieces: self.pieces.clone(),
            words: self.words.clone(),
            turns: self.turns.clone(),
            markdown: self.markdown.clone(),
            latest_transcription_json: self.latest_transcription_json.clone(),
        }
    }

    pub fn spans(&self) -> &[SpeakerSpan] {
        &self.spans
    }

    pub fn pieces(&self) -> &[TranscriptPiece] {
        &self.pieces
    }

    pub fn words(&self) -> &[TranscriptPiece] {
        &self.words
    }

    pub fn turns(&self) -> &[SpeakerTurn] {
        &self.turns
    }

    pub fn markdown(&self) -> &str {
        &self.markdown
    }
}

fn insert_unique_span(spans: &mut Vec<SpeakerSpan>, span: SpeakerSpan) -> bool {
    if spans.iter().any(|existing| {
        existing.speaker == span.speaker
            && existing.start_sample == span.start_sample
            && existing.end_sample == span.end_sample
    }) {
        return false;
    }
    spans.push(span);
    spans.sort_by_key(|item| (item.start_sample, item.end_sample));
    true
}

fn insert_unique_piece(pieces: &mut Vec<TranscriptPiece>, piece: TranscriptPiece) -> bool {
    if pieces.iter().any(|existing| {
        existing.start_sample == piece.start_sample
            && existing.end_sample == piece.end_sample
            && existing.text == piece.text
    }) {
        return false;
    }
    pieces.push(piece);
    pieces.sort_by_key(|item| (item.start_sample, item.end_sample));
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama_bridge::{
        LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_SPAN_COMMIT,
        LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_PIECE_COMMIT,
        LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_RESULT_JSON,
        LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_WORD_COMMIT,
    };

    fn ev(
        kind: i32,
        start_sample: u64,
        end_sample: u64,
        speaker_id: i32,
        text: &str,
    ) -> AudioSessionEventOwned {
        AudioSessionEventOwned {
            seq_no: 0,
            kind,
            flags: 0,
            start_sample,
            end_sample,
            speaker_id,
            item_id: 0,
            text: text.to_string(),
            detail: String::new(),
        }
    }

    #[test]
    fn orchestrator_builds_markdown_from_span_and_piece_events() {
        let mut orch = DiarizedTranscriptOrchestrator::new(16000);
        orch.ingest_event(&ev(
            LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_SPAN_COMMIT,
            0,
            16000,
            1,
            "SPEAKER_01",
        ));
        orch.ingest_event(&ev(
            LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_PIECE_COMMIT,
            0,
            16000,
            -1,
            "Hello there",
        ));
        assert!(orch.markdown().contains("### SPEAKER_01"));
        assert!(orch.markdown().contains("Hello there"));
    }

    #[test]
    fn orchestrator_collects_words_and_json_without_recomputing_turns() {
        let mut orch = DiarizedTranscriptOrchestrator::new(16000);
        assert!(orch.ingest_event(&ev(
            LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_WORD_COMMIT,
            0,
            100,
            -1,
            "hello",
        )));
        assert_eq!(orch.words().len(), 1);

        let json_ev = ev(
            LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_RESULT_JSON,
            0,
            0,
            -1,
            "{\"mode\":\"timeline\"}",
        );
        assert!(!orch.ingest_event(&json_ev));
        assert_eq!(
            orch.snapshot().latest_transcription_json.as_deref(),
            Some("{\"mode\":\"timeline\"}")
        );
    }
}
