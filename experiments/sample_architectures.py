from src.nas.search_space import sample_architecture


def main():
    print("Sampling candidate architectures...\n")

    for i in range(10):
        architecture = sample_architecture()
        print(f"Candidate {i + 1}: {architecture}")


if __name__ == "__main__":
    main()