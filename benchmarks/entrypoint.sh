#!/bin/bash
set -e

# "command" to pass docker run to execute benchmarks
RUN_BENCHMARKS="benchmark"
# "command" to pass docker run to get a shell in the benchmark user environment
BENCHMARK_USER_LOGIN="peek"

if [[ "$1" == "$RUN_BENCHMARKS" || "$1" == "$BENCHMARK_USER_LOGIN" ]]; then
    USER_ID=${DUID:-1000}
    GROUP_ID=${DGUI:-1000}
    GROUP_NAME="benchmark"
    USER_NAME=$GROUP_NAME

    # create group for GROUP_ID if one doesn't already exist
    if ! getent group $GROUP_ID &> /dev/null; then
      groupadd -r $GROUP_NAME -g $GROUP_ID
    fi

    # create user for USER_ID if one doesn't already exist
    if ! getent passwd $USER_ID &> /dev/null; then
      useradd -u $USER_ID -g $GROUP_ID $USER_NAME
    fi

    mkdir /temp-home
    chown -R $USER_ID:$GROUP_ID /temp-home
    
    # modify benchmark user to have /bin/bash as shell
    usermod -s /bin/bash $(id -u -n $USER_ID) -d /temp-home

    sync

    
    if [[ "$1" == "$RUN_BENCHMARKS" ]]; then
      # step-down from root and run benchmarks as benchmark user
      exec pysu $(id -u -n $USER_ID) container-benchmarks/benchmarks/runner.sh
    else
      # step-down from root and run bash (for exploration in interactive mode)
      exec pysu $(id -u -n $USER_ID) /bin/bash
    fi
fi

exec "$@"
