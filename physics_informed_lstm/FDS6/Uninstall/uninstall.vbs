Set objShell = CreateObject("Shell.Application")                       
Set objWshShell = WScript.CreateObject("WScript.Shell")               
Set objWshProcessEnv = objWshShell.Environment("PROCESS")             
objShell.ShellExecute "D:\FDS\\FDS6\Uninstall\uninstall_base.bat", "", "", "runas" 
WScript.Sleep 10000                                                   
