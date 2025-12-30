import editdistance


def get_distance(ref, hyp):
    return editdistance.eval(ref, hyp)


def calc_cer(target_text, predicted_text) -> float:
    """
    Calculate Character Error Rate (CER).
    CER = (S + D + I) / N
    where N is the length of the reference text.
    """
    if not target_text:
        # Если цель пустая, но предсказание есть, это ошибка
        # Если и то и другое пустое, это не ошибка
        return 1.0 if predicted_text else 0.0

    dist = get_distance(target_text, predicted_text)
    return dist / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    """
    Calculate Word Error Rate (WER).
    Same as CER but tokenized by space.
    """
    target_words = target_text.split()
    predicted_words = predicted_text.split()

    if not target_words:
        return 1.0 if predicted_words else 0.0

    dist = get_distance(target_words, predicted_words)
    return dist / len(target_words)
