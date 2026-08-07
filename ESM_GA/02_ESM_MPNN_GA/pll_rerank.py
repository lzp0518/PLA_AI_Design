import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM


WT_SEQUENCE = (
    "MFLRREFGAVAALSVLAHAAPAPAPMQRRDISSTVLDNIDLFAQYSAAAYCSSNIESTGTTLTCDVGN"
    "CPLVEAAGATTIDEFDDTSSYGDPTGFIAVDPTNELIVLSFRGSSDLSNWIADLNFGLTSVSSICDGC"
    "EMHKGFYEAWEVIADTITSKVEAAVSSYPDYTLVFTGHSYGAALAAVAATVLRNAGYTLDLYNFGQPRI"
    "GNLALADYITGQNMGSNYRVTHTDDIVPKLPPELLGYHHFSPEYWITSGNDVTVTTSDVTEVVGVDSTA"
    "GNDGTLLDSTTAHRWYTIYISECS"
)


def compute_pll(seq, tokenizer, model, device, batch_size=32):
    masked_sequences = []

    for pos in range(len(seq)):
        masked_seq = list(seq)
        masked_seq[pos] = tokenizer.mask_token
        masked_sequences.append("".join(masked_seq))

    total_score = 0.0

    for start in range(0, len(masked_sequences), batch_size):
        batch = masked_sequences[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(**inputs).logits
            else:
                logits = model(**inputs).logits

        log_probs = torch.log_softmax(logits.float(), dim=-1)

        for batch_idx in range(len(batch)):
            seq_pos = start + batch_idx
            aa_id = tokenizer.convert_tokens_to_ids(seq[seq_pos])
            total_score += log_probs[
                batch_idx,
                seq_pos + 1,
                aa_id
            ].item()

    return total_score / len(seq)


def rerank_top300(
    top300,
    wt_seq,
    tokenizer,
    model,
    device,
    output_path,
    batch_size=32
):
    required_columns = {"Rank", "Fitness", "Mutations", "Sequence"}
    missing_columns = required_columns.difference(top300.columns)
    if missing_columns:
        raise ValueError(
            "Top300 input is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    invalid_lengths = top300["Sequence"].astype(str).str.len() != len(wt_seq)
    if invalid_lengths.any():
        bad_rows = top300.index[invalid_lengths].tolist()[:10]
        raise ValueError(
            f"Sequence length differs from WT at rows: {bad_rows}"
        )

    print("\nComputing WT full-sequence PLL...")
    wt_pll = compute_pll(
        wt_seq,
        tokenizer,
        model,
        device,
        batch_size=batch_size
    )
    print(f"WT PLL={wt_pll:.4f}")

    print("\nComputing Top300 full-sequence PLL...")
    pll_scores = [
        compute_pll(
            seq,
            tokenizer,
            model,
            device,
            batch_size=batch_size
        )
        for seq in tqdm(top300["Sequence"].astype(str))
    ]

    ranked = top300.copy()
    ranked["GA_Rank"] = ranked["Rank"]
    ranked["WT_PLL"] = wt_pll
    ranked["ESM_PLL"] = pll_scores
    ranked["Delta_PLL"] = ranked["ESM_PLL"] - wt_pll

    delta_min = ranked["Delta_PLL"].min()
    delta_max = ranked["Delta_PLL"].max()
    ranked["Delta_PLL_norm"] = (
        ranked["Delta_PLL"] - delta_min
    ) / (
        delta_max - delta_min + 1e-8
    )

    ranked = ranked.sort_values(
        "ESM_PLL",
        ascending=False
    ).reset_index(drop=True)
    ranked.insert(0, "PLL_Rank", range(1, len(ranked) + 1))
    ranked.to_csv(output_path, index=False)

    return ranked


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Rerank the ESM-ThermoMPNN GA Top300 by full-sequence ESM PLL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "Top300_GA_ESM_ThermoMPNN.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "Top300_ESM_PLL_ranked.csv"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=script_dir.parent / "esm1v"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.model_dir.is_dir():
        raise FileNotFoundError(args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = EsmForMaskedLM.from_pretrained(args.model_dir).to(device)

    if device.type == "cuda":
        model = model.half()

    model.eval()
    top300 = pd.read_csv(args.input)
    ranked = rerank_top300(
        top300,
        WT_SEQUENCE,
        tokenizer,
        model,
        device,
        output_path=args.output,
        batch_size=args.batch_size
    )

    print("\nPLL Top10:")
    print(
        ranked[
            [
                "PLL_Rank",
                "GA_Rank",
                "Fitness",
                "ESM_PLL",
                "Delta_PLL",
                "Mutations"
            ]
        ].head(10)
    )
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
