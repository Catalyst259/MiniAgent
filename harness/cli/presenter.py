"""Owns conversation cell lifecycles independently of the output target."""
from __future__ import annotations

from harness.cli import events as ui
from harness.cli.cells import (
    AssistantCell, ErrorCell, InfoCell, SkillCell, SubAgentCell,
    ToolCell, ToolStatus, UserCell,
)
from harness.cli.output import ConsoleOutput, TranscriptOutput
from harness.cli.state import AppState


class ConversationPresenter:
    def __init__(self, state: AppState, output: ConsoleOutput | TranscriptOutput) -> None:
        self.state = state
        self.output = output
        self._tool_cells: dict[str, ToolCell] = {}
        self._subagent_cell: SubAgentCell | None = None
        self._assistant_cell: AssistantCell | None = None
        self.final_cell: AssistantCell | None = None

    def emit(self, event: ui.AgentEvent) -> None:
        """Consume one AgentEvent: mutate state and cells (render happens after)."""

        cell = None
        if isinstance(event, ui.TurnStarted):
            self._assistant_cell = None
            self.final_cell = None
            self.state.activity = "starting task"
            cell = UserCell(text=event.prompt)
            self.state.append_cell(cell)

        elif isinstance(event, ui.AssistantStarted):
            # A new assistant message: create the cell once and keep updating it
            # for the rest of the stream.  No other path may append another.
            self._drop_empty_assistant_cell()
            self.state.activity = f"iteration {event.iteration}: waiting for model"
            # a finished message that came before this one (text the model wrote
            # alongside tool calls) belongs in the transcript: render it now
            cell = self._assistant_cell
            self._assistant_cell = AssistantCell(message_id=event.message_id)
            self.state.append_cell(self._assistant_cell)

        elif isinstance(event, ui.AssistantDelta):
            self.state.activity = "receiving assistant response"
            if self._assistant_cell is None:
                self._assistant_cell = AssistantCell(message_id=event.message_id)
                self.state.append_cell(self._assistant_cell)
            self._assistant_cell.append_delta(event.text)

        elif isinstance(event, ui.AssistantFinished):
            # Completion only marks the streaming cell done; it never appends or
            # renders a second copy of the same answer.
            self._finalize_pending_assistant(event.text, event.reasoning)
            cell = self._assistant_cell

        elif isinstance(event, ui.ToolStarted):
            # an assistant message that only produced tool calls leaves an empty
            # cell behind; the turn has clearly moved on, so drop it here
            self._drop_empty_assistant_cell()
            self.state.activity = f"running {event.tool}"
            # identity is the call id: "requested" and "started" are the SAME
            # entity, so the second event updates the existing cell instead of
            # creating another one (that produced duplicated pending lines).
            key = event.call_id or f"anon-{len(self._tool_cells)}"
            cell = self._tool_cells.get(key)
            if cell is None:
                cell = ToolCell(call_id=key, tool=event.tool, arguments=event.arguments)
                self._tool_cells[key] = cell
                self.state.set_active(cell)
            else:
                if event.arguments and not cell.arguments:
                    cell.arguments = event.arguments
                if event.tool and not cell.tool:
                    cell.tool = event.tool

        elif isinstance(event, ui.ToolOutput):
            cell = self._tool_cells.get(event.call_id)
            if cell is not None:
                cell.append(event.text)

        elif isinstance(event, ui.ToolFinished):
            cell = self._tool_cells.get(event.call_id)
            if cell is None:
                cell = ToolCell(call_id=event.call_id, tool=event.tool or "tool")
            cell.finish(event.ok, text=event.text, duration_ms=event.duration_ms)
            self.state.commit(cell)
            self.state.activity = "waiting for model"
            self._tool_cells.pop(event.call_id, None)

        elif isinstance(event, ui.ToolFailed):
            cell = self._tool_cells.get(event.call_id)
            if cell is None:
                cell = ToolCell(call_id=event.call_id, tool=event.tool or "tool")
            cell.fail(event.error)
            self.state.commit(cell)
            self.state.activity = "waiting for model"
            self._tool_cells.pop(event.call_id, None)

        elif isinstance(event, ui.SkillLoaded):
            self.state.append_cell(SkillCell(name=event.name, ok=event.ok))

        elif isinstance(event, ui.SubAgentStarted):
            cell = SubAgentCell(agent=event.agent, task=event.task)
            self._subagent_cell = cell
            self.state.set_active(cell)

        elif isinstance(event, ui.SubAgentFinished):
            cell = self._subagent_cell or SubAgentCell(agent=event.agent)
            cell.ok = event.ok
            cell.iterations = event.iterations
            cell.summary = event.summary or cell.summary
            cell.status = ToolStatus.DONE if event.ok else ToolStatus.FAILED
            self.state.commit(cell)
            self._subagent_cell = None

        elif isinstance(event, ui.Compacted):
            if event.folded:
                self.state.append_cell(InfoCell(message=f"◆ compacted {event.folded} message(s)"))

        elif isinstance(event, ui.TurnFinished):
            self.state.activity = ""
            self._finalize_pending_assistant(event.final_answer, None)
            self.final_cell = self._assistant_cell
            cell = self.final_cell
            self._assistant_cell = None

        elif isinstance(event, ui.PermissionDecided):
            self._handle_permission_decision(event)

        elif isinstance(event, ui.InteractionResolved):
            if event.ok:
                self.state.append_cell(
                    InfoCell(message=f"? {event.question} → {event.value}")
                )
            else:
                self.state.append_cell(
                    InfoCell(message=f"? {event.question} — cancelled: {event.error}")
                )

        elif isinstance(event, ui.ErrorEvent):
            self.state.append_cell(ErrorCell(message=event.message, fatal=event.fatal))

        self.output.present(event, self.state, cell)

    def _handle_permission_decision(self, event: ui.PermissionDecided) -> None:
        """Render one permission verdict.

        Denials become transcript cells so the refusal is visible even in a long
        run; approvals are quiet (the tool cell that follows is the record),
        except when the user answered a prompt - then the choice is echoed so the
        transcript shows *who* allowed it.
        """

        if event.permission == "deny":
            label = f"⛔ permission denied · {event.tool or 'tool'}"
            if event.reason:
                label += f" — {event.reason}"
            self.state.append_cell(InfoCell(message=label))
            return

        if event.approval in ("once", "session", "persistent"):
            scope = {
                "once": "once",
                "session": "for this session",
                "persistent": "always",
            }[event.approval]
            self.state.append_cell(
                InfoCell(message=f"⚠ allowed {scope} · {event.tool or 'tool'}")
            )

    def _finalize_pending_assistant(self, text: str | None, reasoning: str | None) -> None:
        cell = self._assistant_cell
        if cell is None:
            return
        if text:
            cell.complete_with(text)
        else:
            cell.complete = True
        cell.reasoning = reasoning or cell.reasoning
        self._drop_empty_assistant_cell()

    def render(self) -> None:
        self.output.refresh(self.state)

    def notify(self, message: str, *, error: bool = False) -> None:
        self.output.notify(self.state, message, error=error)

    def emit_line(self, text: str = "") -> None:
        self.output.notify(self.state, text, line=True)

    def stop_turn(self, reason: str) -> None:
        """Close unfinished cells when execution is cancelled or fails."""
        self.emit(ui.AssistantFinished())
        self._assistant_cell = None
        for call_id in list(self._tool_cells):
            self.emit(ui.ToolFailed(call_id=call_id, error=reason))
        if self._subagent_cell is not None:
            self.emit(ui.SubAgentFinished(agent=self._subagent_cell.agent, ok=False, summary=reason))
        self.state.activity = ""
        self.final_cell = None
        self.render()

    def clear(self) -> None:
        self._tool_cells.clear()
        self._subagent_cell = None
        self._assistant_cell = None
        self.final_cell = None
        self.state.history_cells.clear()
        self.state.active_cell = None
        self.state.expanded_tool_ids.clear()
        self.state.activity = ""
        self.output.clear()
        self.state.touch()

    def _drop_empty_assistant_cell(self) -> None:
        """Discard a streamed message that turned out to be empty.

        A turn whose model answer is only tool calls leaves an empty assistant
        cell behind; without this it would sit in the transcript (and be counted)
        for the rest of the session.
        """

        cell = self._assistant_cell
        if cell is None or not cell.empty:
            return
        if self.state.history_cells and self.state.history_cells[-1] is cell:
            self.state.history_cells.pop()
        self._assistant_cell = None
