#!/bin/bash

# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
# Systems

# Process command-line arguments.
set -u
while [[ $# > 1 ]]; do
  case $1 in
    --username )
      username=$2
      ;;
    --private_ssh_key_path )
      private_ssh_key_path=$2
      ;;
    --node_0 )
      node_0=$2
      ;;
    --node_1 )
      node_1=$2
      ;;
    --node_2 )
      node_2=$2
      ;;
    --node_3 )
      node_3=$2
      ;;
    --node_4 )
      node_4=$2
      ;;
    --node_5 )
      node_5=$2
      ;;
    * )
      echo "Invalid argument: $1"
      exit 1
  esac
  shift
  shift
done

# Copy the SSH private key to node-0.
scp -o StrictHostKeyChecking=no ${private_ssh_key_path} ${username}@${node_0}:.ssh/id_rsa

# Log into node-0.
ssh -o StrictHostKeyChecking=no ${username}@${node_0} "
  # Install Docker.
  sudo apt-get update
  sudo apt-get -y install apt-transport-https ca-certificates curl gnupg-agent \
      software-properties-common
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
  sudo add-apt-repository \"deb [arch=amd64] https://download.docker.com/linux/ubuntu \$(lsb_release -cs) stable\"
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io

  # Clone this repository to get the experiment configuration files.
  ssh-keygen -F github.com || ssh-keyscan github.com >> ~/.ssh/known_hosts
  git clone git@github.com:rodrigoalveslima/BuzzBlogBenchmark.git
  mv BuzzBlogBenchmark/controller/conf/* .
  rm -rf BuzzBlogBenchmark

  # Replace variables in system.yml.
  sed -i \"s/{{username}}/${username}/g\" system.yml
  sed -i \"s/{{node-0}}/${node_0}/g\" system.yml
  sed -i \"s/{{node-1}}/${node_1}/g\" system.yml
  sed -i \"s/{{node-2}}/${node_2}/g\" system.yml
  sed -i \"s/{{node-3}}/${node_3}/g\" system.yml
  sed -i \"s/{{node-4}}/${node_4}/g\" system.yml
  sed -i \"s/{{node-5}}/${node_5}/g\" system.yml
"
