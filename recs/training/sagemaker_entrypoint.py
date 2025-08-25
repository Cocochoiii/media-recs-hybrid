# Example SageMaker training entrypoint (skeleton)
import os, argparse, json
from recs.training.train_mf import main as train_mf_main
from recs.training.train_content import main as train_content_main

if __name__ == "__main__":
    # Typically SM passes env vars for data/model channels; adapt as needed.
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--epochs", type=int, default=3)
    args, _ = parser.parse_known_args()

    # In a real job, you'd load data from /opt/ml/input/data/
    train_mf_main(argparse.Namespace(factors=64, lr=1e-2, weight_decay=1e-5, epochs=args.epochs, neg_ratio=3, batch_size=2048, out_dir=args.out_dir))
    train_content_main(argparse.Namespace(out_dir=args.out_dir))

    with open(os.path.join(args.out_dir, "done.json"), "w") as f:
        json.dump({"status": "ok"}, f)
