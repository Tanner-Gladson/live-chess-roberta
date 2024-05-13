
import chess
import random
import torch
from typing import Any, Optional, Union
import logging
import pandas as pd
import sys
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForMaskedLM,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    pipeline,
    BatchEncoding,
)

logger = logging.getLogger(__name__)


class ChessRobertaDriver:
  '''A wrapper for the ChessRoberta huggingface model'''
  LONGEST_POSSIBLE_SAN = 7
  MAX_SANS_FOR_INFERENCE_GAME = 6 # The number of half-moves to be kept in the target_frame during inference
  BATCH_SIZE = 16 # The number of possible moves to sameple from the model
  MODEL_SEQ_LENGTH = 1024

  SAMPLES_TEXT_PATH = "data/random_frames.txt" # bad practice but on a time crunch
  SPECIAL_TOKENS = {
    "PGN_START": "~",
    "MOVE_SEP": ">",
    "PGN_PADING": "_",
  }

  def __init__(self):
    logger.info("Initializing a ChessRobertaDriver instance")
    # if torch.cuda.is_available():
    #   logger.info("using cuda")
    #   self.device = torch.device("cuda")
    # else:
    #   logger.info("using cpu")
    #   self.device = torch.device("cpu")
  
    # self.tokenizer = AutoTokenizer.from_pretrained("TannerGladson/chess-roberta")
    # self.model = AutoModelForMaskedLM.from_pretrained("TannerGladson/chess-roberta")
    # self.model.to(self.device)

    # # we need a small corpus to sample from for packing sequences
    # sample_frames : list[str] = ChessRobertaDriver.get_samples()
    # self.sample_frame_encodings : list[list[int]] = [self.tokenizer(frame, return_tensors="pt")["input_ids"].tolist() for frame in sample_frames]
    logger.info("ChessRobertaDriver parameters have been loaded")

  def get_next_play(self, moves: list[str], board: chess.Board) -> chess.Move:
    '''Return the next UCI move in a game. Assumes moves contains all moves from starting FEN
    Parameters:
      - moves: a list of UCI moves from the starting FEN
    '''
    generated_moves : list[chess.Move] = self.generate_move_candidates(moves)
    all_legal_moves = list(board.legal_moves)
    legal_generated_moves = [move for move in generated_moves if move in all_legal_moves]

    if len(legal_generated_moves) == 0:
      logger.info("ChessRobertaDriver did not generate any legal moves, choosing random move")
      return random.choice(all_legal_moves)
    
    return random.choice(legal_generated_moves)
  
  def generate_move_candidates(self, moves: list[str]) -> list[chess.Move]:
    ''' Generate a batch of candidates for the next move'''
    # # generate a ChessRoberta style input which begins with this game
    # target_frame : str = ChessRobertaDriver.generate_chess_roberta_style_frame(moves)
    # batched_frames : BatchEncoding = self.generate_tokenized_batch(target_frame)

    # # now mask out the last 7 tokens in the target frame. Don't forget a [CLS] and [SEP] were added
    # target_encoding_len = len(target_frame) + 2
    # start_of_mask = target_encoding_len - ChessRobertaDriver.LONGEST_POSSIBLE_SAN - 1

    # mask = torch.zeros(ChessRobertaDriver.BATCH_SIZE, target_encoding_len, dtype=torch.bool, device=self.device)
    # mask = mask[:, start_of_mask:start_of_mask+7] = True

    # print("mask:")
    # print(mask)

    # batched_frames['input_ids'][mask] = self.tokenizer.mask_token_id
    # batched_frames.to(self.device)

    # with torch.no_grad():
    #   logits = self.model(**batched_frames).logits

    # decoded_logits = self.tokenizer.decode(logits.argmax(axis=-1)[0])
    # # TODO: parse those tokens to identify at most the next move

    # # TODO: convert this move from PGN to UCI using the chess library
    
    return []

  def generate_tokenized_batch(self, target_frame: str) -> BatchEncoding:
    '''Generate BATCH_SIZE tokenized sequences where target_frame is the first frame in each sample'''
    CONTEXT_WINDOW_SIZE = ChessRobertaDriver.MODEL_SEQ_LENGTH
    
    target_frame_encoding = self.tokenizer(target_frame, return_tensors="pt")["input_ids"].tolist()
    self.sample_frame_encodings

    batch = []
    for _ in range(ChessRobertaDriver.BATCH_SIZE):
      sequence = []
      sequence.extend(target_frame_encoding)

      # Each sequence is comprised by a concatenation of random samples
      random_sample = random.choice(self.sample_frame_encodings)
      while len(sequence) + len(random_sample) < CONTEXT_WINDOW_SIZE:
        sequence.extend(random_sample)
        random_sample = random.choice(self.sample_frame_encodings)
      
      # With a padding at the end
      deficit = CONTEXT_WINDOW_SIZE - len(sequence)
      sequence.extend([self.tokenizer.pad_token_id] * deficit)
      batch.append(sequence)

    # the model expects a few properties for each batch
    result = {
      "input_ids": batch,
      "attention_mask": [[1] * CONTEXT_WINDOW_SIZE for _ in range(ChessRobertaDriver.BATCH_SIZE)], # 1024 x BATCH_SIZE
      "token_type_ids": [[0] * CONTEXT_WINDOW_SIZE for _ in range(ChessRobertaDriver.BATCH_SIZE)], # 1024 x BATCH_SIZE
    }

    # Tensorize the results
    for key in result.keys():
      result[key] = torch.tensor(result[key])
    return BatchEncoding(result)
  
  @staticmethod
  def generate_chess_roberta_style_frame(moves: list[str]) -> str:
    num_moves_to_keep = min(len(moves), ChessRobertaDriver.MAX_SANS_FOR_INFERENCE_GAME)
    
    board = chess.Board()
    for move in moves[:-num_moves_to_keep]:
      board.push_uci(move)
    fen_to_prepend = board.fen()

    san_moves = []
    for move in moves[-num_moves_to_keep:]:
      board.push_uci(move)
      san = board.san(move)
      san_moves.append(san)

    pgn_start_tok = ChessRobertaDriver.SPECIAL_TOKENS["PGN_START"]
    move_sep_tok = ChessRobertaDriver.SPECIAL_TOKENS["MOVE_SEP"]
    padding_tok = ChessRobertaDriver.SPECIAL_TOKENS["PGN_PADING"]
    num_pad_toks = ChessRobertaDriver.LONGEST_POSSIBLE_SAN
    target_frame : str = fen_to_prepend + pgn_start_tok + move_sep_tok.join(san_moves) + move_sep_tok + padding_tok*num_pad_toks
    return target_frame

  @staticmethod
  def get_samples() -> list[str]:
    '''Get up to N random samples from a dataset'''
    samples = []
    with open(ChessRobertaDriver.SAMPLES_TEXT_PATH, 'r') as file:
      for line in file:
        samples.append(line[:-1]) # remove newline character
    return samples