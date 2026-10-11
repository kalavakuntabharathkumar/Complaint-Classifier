import argparse, statistics, time
from .onnx_predict import ComplaintPredictor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="I was charged a fee that I do not recognize on my account.")
    parser.add_argument("--runs", type=int, default=1000)
    args = parser.parse_args()
    predictor = ComplaintPredictor()
    for _ in range(20):
        predictor.predict(args.text)
    samples = []
    for _ in range(args.runs):
        start = time.perf_counter()
        predictor.predict(args.text)
        samples.append((time.perf_counter() - start) * 1000)
    print(f"runs={args.runs} median_ms={statistics.median(samples):.2f} p95_ms={sorted(samples)[int(.95*len(samples))-1]:.2f}")

if __name__ == "__main__":
    main()
