"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""

from __future__ import annotations
import chess
import random
from lib.engine_wrapper import MinimalEngine, MOVE
from typing import Any, Optional, Union
import logging
import sys
from lib.config import Configuration
from lib import model

OPTIONS_TYPE = dict[str, Any]
COMMANDS_TYPE = list[str]
MOVE = Union[chess.engine.PlayResult, list[chess.Move]]
logger = logging.getLogger(__name__)


class ChessRobertaDriver:
    '''A wrapper for the ChessRoberta huggingface model'''
    def __init__(self):
        print("Initializing a ChessRobertaDriver instance")

    def get_next_play(game: chess.Game) -> chess.engine.PlayResult:
        pass


class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""
    pass


class ChessRobertaEngine(ExampleEngine):
    """Chess Roberta! But for now, get a random move."""

    def __init__(self, commands: COMMANDS_TYPE, options: OPTIONS_TYPE, stderr: Optional[int],
                draw_or_resign: Configuration, game: Optional[model.Game], **popen_args: str):
        """Start Stockfish."""
        super().__init__(commands, options, stderr, draw_or_resign, game, **popen_args)
        self.chess_roberta_driver = ChessRobertaDriver()


    def search(self, board: chess.Board, game: model.Game, *args: Any) -> chess.engine.PlayResult:
        """Choose a random move."""
        # TODO: Compress all information into a chess.Game object
        full_game : chess.Game = None

        # return self.chess_roberta_driver.get_next_play(full_game)
        return chess.engine.PlayResult(random.choice(list(board.legal_moves)), None)