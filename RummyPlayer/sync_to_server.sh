ai#!/bin/bash

# Create the directory on the AI server through the CSCI server
ssh jmm2103@csci.hsutx.edu "ssh jmm2103@ai.hsutx.edu 'mkdir -p /home/jmm2103/RummyPlayer'"

# Sync files to CSCI server first
rsync -av --exclude='.git' --exclude='.vscode' --exclude='__pycache__' --exclude='*.pyc' --exclude='venv' --exclude='sync_to_server.sh' ./ jmm2103@csci.hsutx.edu:/home/jmm2103/temp_rummy/

# SSH into CSCI server and sync files to AI server
ssh jmm2103@csci.hsutx.edu "rsync -av /home/jmm2103/temp_rummy/ jmm2103@ai.hsutx.edu:/home/jmm2103/RummyPlayer/ && rm -rf /home/jmm2103/temp_rummy"
