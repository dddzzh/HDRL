"""Suspend/resume only the identity-checked HEFT evaluation tree, once each."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'results/shared_heft_eval_sm74_20261024/acceleration_12/pause_state.json'
PARENT = 48112
CHILDREN = {60720,30828,58968,39868,23244,57916,46148,41684,60968,61404,8244,2768}


def processes():
    command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } | Select-Object ProcessId,ParentProcessId,@{N='Created';E={$_.CreationDate.ToString('o')}},CommandLine | ConvertTo-Json -Depth 3 -Compress"
    result = subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,check=True)
    value = json.loads(result.stdout)
    return {r['ProcessId']:r for r in (value if isinstance(value,list) else [value])}


def validate_tree(actual, expected=None):
    parent = actual.get(PARENT)
    if parent is None or 'tools/accelerate_shared_heft_sm74.py' not in parent['CommandLine'] or not parent['Created'].startswith('2026-10-24T23:31:13.'):
        raise ValueError('Parent identity changed; no processes touched')
    children = {p for p,r in actual.items() if r['ParentProcessId']==PARENT}
    if children != CHILDREN or any('multiprocessing.spawn' not in actual[p]['CommandLine'] for p in children):
        raise ValueError('Child identities changed; no processes touched')
    tree = {p:actual[p] for p in [PARENT,*sorted(CHILDREN)]}
    if expected is not None and tree != {int(p):r for p,r in expected.items()}:
        raise ValueError('PID reuse/command/creation-time mismatch; no processes touched')
    return tree


def control(action):
    if action=='pause':
        if STATE.exists():
            raise FileExistsError('Pause already recorded; refuse duplicate suspension')
        state = {'status':'pausing','processes':validate_tree(processes()),'pause_returns':{},'resume_returns':{},
                 'resume_not_before':'2026-10-25T06:00:00+08:00'}
    else:
        state = json.loads(STATE.read_text(encoding='utf-8'))
        if state['status'] != 'suspended' or state['resume_returns']:
            raise ValueError('Not a clean suspended state; refuse duplicate resume')
        if datetime.now(timezone.utc) < datetime.fromisoformat(state['resume_not_before']):
            raise ValueError('Resume time has not arrived')
        validate_tree(processes(),state['processes'])
        state['status']='resuming'
    def save():
        STATE.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    native = ctypes.WinDLL('ntdll')
    operation = native.NtSuspendProcess if action=='pause' else native.NtResumeProcess
    operation.argtypes=[wintypes.HANDLE]
    operation.restype=ctypes.c_long
    handles={}
    try:
        for pid in [PARENT,*sorted(CHILDREN)]:
            handle=kernel.OpenProcess(0x0800,False,pid)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            handles[pid]=handle
        save()
        order=[PARENT,*sorted(CHILDREN)] if action=='pause' else [*sorted(CHILDREN),PARENT]
        for pid in order:
            code=int(operation(handles[pid]))
            state[action+'_returns'][str(pid)]=code
            save()
            if code != 0:
                raise RuntimeError(f'{action} failed for {pid}: NTSTATUS={code}')
        state['status']='suspended' if action=='pause' else 'resumed'
        state[action+'_at_utc']=datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps({'status':state['status'],'processes':len(handles),'returns':state[action+'_returns']}))
    finally:
        for handle in handles.values():
            kernel.CloseHandle(handle)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('pause','resume'))
    control(parser.parse_args().action)
