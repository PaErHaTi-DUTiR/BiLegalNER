import datasets

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = "BiLegalNERD"

class BiLegalNERDConfig(datasets.BuilderConfig):
    """BuilderConfig for BiLegalNERD"""

    def __init__(self, **kwargs):
        """BuilderConfig for BiLegalNERD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(BiLegalNERDConfig, self).__init__(**kwargs)


class BiLegalNERD(datasets.GeneratorBasedBuilder):
    """BiLegalNERD dataset."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            supervised_keys=None,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-NASI",
                                "I-NASI",
                                "B-NCGV",
                                "I-NCGV",
                                "B-NCSM",
                                "I-NCSM",
                                "B-NHCS",
                                "I-NHCS",
                                "B-NHVI",
                                "I-NHVI",
                                "B-NO",
                                "I-NO",
                                "B-NS",
                                "I-NS",
                                "B-NT",
                                "I-NT",
                                "B-NATS",
                                "I-NATS",
                                "B-NCSP",
                                "I-NCSP"
                            ]
                        )
                    )
                }
            )
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={
                    "filepath": './BiLegalNERD-train.txt'
                }),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={
                    "filepath": './valid-ug.txt'
                }),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={
                    "filepath": './test-ug.txt'
                }),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        valid_labels = set([
            "O",
            "B-NASI",
            "I-NASI",
            "B-NCGV",
            "I-NCGV",
            "B-NCSM",
            "I-NCSM",
            "B-NHCS",
            "I-NHCS",
            "B-NHVI",
            "I-NHVI",
            "B-NO",
            "I-NO",
            "B-NS",
            "I-NS",
            "B-NT",
            "I-NT",
            "B-NATS",
            "I-NATS",
            "B-NCSP",
            "I-NCSP"
        ])
        
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            try:
                for line in f:
                    if line.strip() == "":
                        if tokens:
                            yield guid, {
                                "id": str(guid),
                                "tokens": tokens,
                                "ner_tags": ner_tags,
                            }
                            guid += 1
                            tokens = []
                            ner_tags = []
                    else:
                        splits = line.split()
                        if len(splits) >= 2:  # Ensure there are at least two elements
                            token, ner_tag = splits[0], splits[1].rstrip()
                            if ner_tag not in valid_labels:
                                logger.warning(f"Skipping token with invalid label: {token} - {ner_tag}")
                                continue
                            tokens.append(token)
                            ner_tags.append(ner_tag)
                        else:
                            logger.warning(f"Skipping line with format error: {line.strip()}")
                
                if tokens:  # Ensure the last example is yielded
                    yield guid, {
                        "id": str(guid),
                        "tokens": tokens,
                        "ner_tags": ner_tags,
                    }
            except Exception as e:
                logger.error(f"An error occurred while processing the file: {e}")
                raise

    def _prepare_split(self, split_generator, **prepare_split_kwargs):
        try:
            super()._prepare_split(split_generator, **prepare_split_kwargs)
        except Exception as e:
            logger.error(f"An error occurred during _prepare_split: {e}")
            raise

    def _prepare_split_single(self, *args, **kwargs):
        try:
            return super()._prepare_split_single(*args, **kwargs)
        except Exception as e:
            logger.error(f"An error occurred during _prepare_split_single: {e}")
            raise

def select_tokens(unique_tokens, total_percentage):
    """
    :param unique_tokens: 一个包含所有唯一词汇的列表
    :param total_percentage: 要抽取的总词汇的百分比
    :return: 抽取的词汇列表
    """
    portion_size = len(unique_tokens) // 10
    selected_tokens = []

    for i in range(10):
        start_index = i * portion_size
        end_index = (i + 1) * portion_size

        if i == 9:
            end_index = len(unique_tokens)

        extract_count = int((end_index - start_index) * (total_percentage / 100.0))

        selected_tokens.extend(unique_tokens[start_index:start_index + extract_count])

    return selected_tokens
