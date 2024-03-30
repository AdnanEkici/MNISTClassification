#!/bin/bash

venv_path="$1"

RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
BLUE='\033[34m'
PURPLE='\033[35m'
RESET='\033[0m'

# Function to confirm user action
confirm_action() {
    while true; do
        read -r -p "Do you want to create a virtual environment and install requirements in '$venv_path'? (yes/no): " choice
        case "$choice" in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo -e "${PURPLE}Please answer 'yes' or 'no'.${RESET}";;
        esac
    done
}

# Ask the user for confirmation
if confirm_action; then
    if ! command -v python3 >/dev/null 2>&1; then
        echo -e "${RED}Error: 'python3' is not installed. Please install Python 3.${RESET}\n"
        exit 1
    fi

    if [ ! -d "$venv_path" ]; then
        echo -e "${BLUE}Creating virtual environment...${RESET}\n"
        python3 -m venv "$venv_path"
    fi

    python_venv_interpreter="$venv_path/bin/python"

    echo -e "${BLUE}Installing requirements...${RESET}\n"

    if "$python_venv_interpreter" -m pip install -r requirements.txt; then
        echo -e "${GREEN}Requirements successfully installed.${RESET}\n"
    else
        echo -e "${RED}Pip install failed. Upgrading pip...${RESET}\n"
        "$python_venv_interpreter" -m pip install --upgrade pip
        echo -e "${BLUE}Retrying installation...${RESET}\n"
        "$python_venv_interpreter" -m pip install -r requirements.txt
    fi

    echo -e "${YELLOW}Warning! Remember to activate the virtual environment.${RESET}\n"
else
    echo -e "${BLUE}Aborted. No action taken.${RESET}"
fi