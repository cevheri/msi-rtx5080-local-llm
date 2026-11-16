#!/bin/bash
echo "### profile use lc"; hermes profile use lc; echo rc=$?; echo "active_profile=$(cat $HOME/.hermes/active_profile 2>&1)"
echo "### plain config get (no -p, HERMES_HOME unset)"; hermes config get model.provider; hermes config path
echo "### HERMES_HOME=$HOME/.hermes explicit"; HERMES_HOME=$HOME/.hermes hermes config path
echo "### as supervised child (HERMES_SUPERVISED_CHILD=1, like the gateway unit)"; HERMES_HOME=$HOME/.hermes HERMES_SUPERVISED_CHILD=1 hermes config path
echo "### -p default"; hermes -p default config path
echo "### profile use default"; hermes profile use default; echo rc=$?; ls -la $HOME/.hermes/active_profile 2>&1 | cut -c1-120; hermes config path
