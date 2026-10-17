"""Tarefas em segundo plano (importacao, exportacao, treino) com progresso
consultavel pelo navegador. Em memoria: some se o servidor reiniciar --
suficiente pra um app local de um usuario/poucos usuarios."""
import os
import tempfile
import threading
import time
import traceback
import uuid

_jobs = {}
_lock = threading.Lock()
JOB_TTL_SECONDS = 6 * 3600


def start(user, work, filename=None, title=""):
    """Roda `work(report)` numa thread. report(frac, msg=None) atualiza o
    progresso. `work` devolve um caminho de arquivo (download) ou um dict
    (resultado JSON). Nao pode tocar em request -- so closure."""
    _cleanup()
    job_id = uuid.uuid4().hex
    job = {"owner": user.id, "progress": 0.0, "message": title, "done": False, "error": None,
           "path": None, "result": None, "filename": filename, "created": time.time()}
    with _lock:
        _jobs[job_id] = job

    def report(frac, msg=None):
        job["progress"] = max(0.0, min(1.0, float(frac)))
        if msg is not None:
            job["message"] = msg

    def run():
        from django.db import connection
        try:
            out = work(report)
            if isinstance(out, str):
                job["path"] = out
            else:
                job["result"] = out
            job["progress"] = 1.0
        except Exception as e:
            traceback.print_exc()
            job["error"] = str(e) or e.__class__.__name__
        finally:
            job["done"] = True
            connection.close()   # thread propria: fecha a conexao do sqlite dela

    threading.Thread(target=run, daemon=True).start()
    return job_id


def get(user, job_id):
    with _lock:
        job = _jobs.get(job_id)
    if not job or job["owner"] != user.id:
        return None
    return job


def pop(job_id):
    with _lock:
        return _jobs.pop(job_id, None)


def _cleanup():
    now = time.time()
    with _lock:
        for jid in [j for j, job in _jobs.items() if job["done"] and now - job["created"] > JOB_TTL_SECONDS]:
            job = _jobs.pop(jid)
            if job.get("path") and os.path.exists(job["path"]):
                try:
                    os.unlink(job["path"])
                except OSError:
                    pass


def temp_file(suffix, write):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as fh:
        write(fh)
    return path
