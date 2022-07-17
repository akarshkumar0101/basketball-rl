mkdir data
cd data

wget https://github.com/linouk23/NBA-Player-Movements/archive/refs/heads/master.zip
unzip master.zip
mv NBA-Player-Movements-master/data/2016.NBA.Raw.SportVU.Game.Logs/* .
rm -rf master.zip NBA-Player-Movements-master

7za x "*.7z"


