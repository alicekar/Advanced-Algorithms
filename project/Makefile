all:
	echo "#!/bin/sh\npython3 main.py\n" > project
	chmod 755 project

clean:
	find . -type f | xargs touch
	rm -f project
