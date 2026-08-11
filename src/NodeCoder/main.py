import argparse


def main():
    parser = argparse.ArgumentParser(description="NodeCoder command line interface")
    parser.add_argument(
        "mode",
        choices=["preprocess", "graph-data", "train", "predict"],
        help="Pipeline step to run",
    )
    parser.add_argument(
        "--protein-id",
        default="NA",
        help="Protein ID used for predict mode (example: KI3L1_HUMAN)",
    )
    args = parser.parse_args()

    if args.mode == "preprocess":
        from NodeCoder import preprocess_raw_data

        preprocess_raw_data.main()
    elif args.mode == "graph-data":
        from NodeCoder import generate_graph_data

        generate_graph_data.main()
    elif args.mode == "train":
        from NodeCoder import train

        train.main()
    elif args.mode == "predict":
        from NodeCoder import predict

        predict.main(protein_ID=args.protein_id)
