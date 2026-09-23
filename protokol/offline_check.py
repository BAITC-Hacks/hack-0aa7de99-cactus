"""Доказательство, что решение не ходит в сеть.

Ограничение кейса: «передача аудио/текста во внешние облачные API
запрещена». Заявить это мало — проверяем принудительно.

Скрипт запрещает процессу любые сетевые соединения на уровне сокетов
и прогоняет весь пайплайн. Если какой-нибудь компонент попытается
обратиться наружу, он получит исключение, и мы увидим, кто именно.

Запуск (веса моделей должны быть уже скачаны):
    python offline_check.py запись.m4a --roster roster.json
"""
import socket, sys, os

BLOCKED: list[str] = []


def cut_network():
    """Глушим сокеты. Локальные адреса оставляем: некоторые библиотеки
    открывают петлю на себя, и это не выход наружу."""
    real_connect = socket.socket.connect
    real_create = socket.create_connection

    def _local(addr) -> bool:
        host = addr[0] if isinstance(addr, (tuple, list)) else str(addr)
        return str(host) in ("127.0.0.1", "::1", "localhost", "0.0.0.0")

    def guard_connect(self, addr, *a, **kw):
        if not _local(addr):
            BLOCKED.append(str(addr))
            raise OSError(f"СЕТЬ ЗАПРЕЩЕНА: попытка соединения с {addr}")
        return real_connect(self, addr, *a, **kw)

    def guard_create(addr, *a, **kw):
        if not _local(addr):
            BLOCKED.append(str(addr))
            raise OSError(f"СЕТЬ ЗАПРЕЩЕНА: попытка соединения с {addr}")
        return real_create(addr, *a, **kw)

    socket.socket.connect = guard_connect
    socket.create_connection = guard_create


if __name__ == "__main__":
    # HuggingFace должен брать веса только из локального кэша
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    cut_network()

    print("Сеть заблокирована на уровне сокетов. Запускаю пайплайн…\n")
    sys.argv = ["run.py"] + sys.argv[1:]
    import run
    try:
        run.main()
        ok = True
    except Exception as e:
        ok = False
        print(f"\nПАЙПЛАЙН УПАЛ: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    if BLOCKED:
        print("ПРОВЕРКА НЕ ПРОЙДЕНА — были попытки выйти в сеть:")
        for a in dict.fromkeys(BLOCKED):
            print("   ", a)
    elif ok:
        print("ПРОВЕРКА ПРОЙДЕНА: протокол собран, ни одного сетевого")
        print("соединения не было. Решение работает в закрытом контуре.")
    else:
        print("Пайплайн не завершился — смотрите ошибку выше.")
    sys.exit(0 if ok and not BLOCKED else 1)
