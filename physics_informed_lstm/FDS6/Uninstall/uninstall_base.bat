@echo off

NET SESSION >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo *** Error: This script is running as %username%.  It must run as Administrator.
    echo       Run again, after right clicking on this script and selecting "Run as Adminstrator"
    echo       FDS/Smokeview uninstaller aborted.
    pause
    exit
)

call :is_cfast_installed
if %cfastinstalled% == 1 goto skip1
  echo *** Removing the association between .smv and Smokeview
  assoc .smv=
  ftype smvDoc=
:skip1

echo *** Removing FDS from the Start menu.
rmdir /q /s "%ALLUSERSPROFILE%\Start Menu\Programs\FDS6"

echo *** Stopping smokeview
taskkill /F /IM smokeview.exe     >Nul 2>Nul

echo *** Stopping fds
taskkill /F /IM fds.exe           >Nul 2>Nul

echo *** Stopping mpiexec
taskkill /F /IM mpiexec.exe       >Nul 2>Nul

echo *** Removing hydra_service
taskkill /F /IM hydra_service.exe >Nul 2>Nul

echo *** Removing smpd
smpd -remove                      >Nul 2>Nul
if "%cfastinstalled%" == "1" goto skip2                
echo Removing "D:\FDS\\SMV6" from the System Path              
call "D:\FDS\\FDS6\Uninstall\set_path.exe" -s -b -r D:\FDS\\SMV6       
rmdir /s /q "D:\FDS\\SMV6"                                     
:skip2                                                   
echo Removing CMDfds desktop shortcut                                   
if exist "C:\Users\manid\Desktop\CMDfds.lnk"   erase "C:\Users\manid\Desktop\CMDfds.lnk"    
if exist "C:\Users\manid\OneDrive\Desktop\CMDfds.lnk" erase "C:\Users\manid\OneDrive\Desktop\CMDfds.lnk"  
echo Removing "D:\FDS\\FDS6\bin" from the System Path          
call "D:\FDS\\FDS6\Uninstall\set_path.exe" -s -b -r "D:\FDS\\FDS6\bin" 
echo.                                                    
echo Removing "D:\FDS\\FDS6"                                   
rmdir /s /q  "D:\FDS\\FDS6"                                    
if exist "D:\FDS\\cfast" goto skip_remove                      
  echo Removing "D:\FDS\\SMV6"                                 
  rmdir /s /q  "D:\FDS\\SMV6"                                  
  echo Removing "D:\FDS\"                           
  rmdir "D:\FDS\"                                   
:skip_remove                                             
echo *** Uninstall complete                              
goto eof

:is_cfast_installed
cfast 1> %temp%\file_exist.txt 2>&1
type %temp%\file_exist.txt | find /i /c "not recognized" > %temp%\file_exist_count.txt
set /p nothave=<%temp%\file_exist_count.txt
set cfastinstalled=1
if %nothave% == 1 (
  set cfastinstalled=0
)
exit /b 0

:eof
pause
