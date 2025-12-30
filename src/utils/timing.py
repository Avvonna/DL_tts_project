import functools
import time


def log_examples_per_sec(get_n_examples, get_mode):
    """
    Логирует examples_per_sec = n_examples / dt.
    get_n_steps: callable(self, *args, **kwargs) -> int (число примеров)
    """

    def deco(fn):
        @functools.wraps(fn)
        def wrapped(self, *args, **kwargs):
            t0 = time.perf_counter()
            out = fn(self, *args, **kwargs)

            writer = getattr(self, "writer", None)
            if writer is not None:
                dt = time.perf_counter() - t0
                n_steps = int(get_n_examples(self, *args, **kwargs))
                mode = str(get_mode(self, *args, **kwargs))

                # гарантируем суффикс _train / _val и т.п.
                writer.mode = mode

                if dt > 0 and n_steps > 0:
                    writer.add_scalar("examples_per_sec", float(n_steps / dt))

            return out

        return wrapped

    return deco
