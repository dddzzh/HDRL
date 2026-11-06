$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$python = 'D:\Miniconda\envs\py310\python.exe'
$logRoot = Join-Path $repo 'results\scenario_training_logs_20260924'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$waves = @(
    @(
        @{ Name = 's37_ours';  Tool = 'tools\train_coupled_hierarchical_d3qn_sac.py'; Config = 'configs\v2_small_28edge_9cloud_dvfs_v3_future_esp.json' },
        @{ Name = 's37_dqn';   Tool = 'tools\train_dqn_edge.py'; Config = 'configs\dqn_edge_small_28edge_9cloud.json' },
        @{ Name = 's37_dream'; Tool = 'tools\train_dream.py'; Config = 'configs\dream_dqn_small_28edge_9cloud.json' },
        @{ Name = 's37_ooac';  Tool = 'tools\train_oo_actor_critic.py'; Config = 'configs\oo_actor_critic_small_28edge_9cloud.json' }
    ),
    @(
        @{ Name = 'm53_ours';  Tool = 'tools\train_coupled_hierarchical_d3qn_sac.py'; Config = 'configs\v2_medium_40edge_13cloud_dvfs_v3_future_esp.json' },
        @{ Name = 'm53_dqn';   Tool = 'tools\train_dqn_edge.py'; Config = 'configs\dqn_edge_medium_40edge_13cloud.json' },
        @{ Name = 'm53_dream'; Tool = 'tools\train_dream.py'; Config = 'configs\dream_dqn_medium_40edge_13cloud.json' },
        @{ Name = 'm53_ooac';  Tool = 'tools\train_oo_actor_critic.py'; Config = 'configs\oo_actor_critic_medium_40edge_13cloud.json' }
    )
)

$status = @()
for ($waveIndex = 0; $waveIndex -lt $waves.Count; $waveIndex++) {
    $running = @()
    foreach ($job in $waves[$waveIndex]) {
        $stdout = Join-Path $logRoot "$($job.Name).stdout.log"
        $stderr = Join-Path $logRoot "$($job.Name).stderr.log"
        $process = Start-Process -FilePath $python `
            -ArgumentList @($job.Tool, '--config', $job.Config) `
            -WorkingDirectory $repo -WindowStyle Hidden `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        $running += @{ Job = $job; Process = $process }
    }

    foreach ($item in $running) {
        $item.Process.WaitForExit()
        $status += [pscustomobject]@{
            wave = $waveIndex + 1
            name = $item.Job.Name
            pid = $item.Process.Id
            exit_code = $item.Process.ExitCode
            finished_at = (Get-Date).ToString('s')
        }
        $status | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logRoot 'status.json')
    }
}
