# sendMail
A simple command-line tool that uses cURL since I always mess up the mailx configurations.
Note that this tools works only for gmail at the moment.
A nice way of using this tool is to execute long experiments and send a mail when finished.
For instance:

	$ ./longExperiment && ./sendMail -x -t "recevier's Mail" -s "subejct" -m "message"

# How to use

As a first step you need to configure you account settings by using the following command:

	$ ./sendMail -c -a "youEmailaccount@gmail.com" -p "youEmailaccountsPassword"

If the above step is successful, then you can start sending emails:

	$ ./sendMail -x -t "receiver_1, receiver_2, receiver_N" -s "emailSubject" -m "emailBody"

You can also add the executable `sendMail` file under the `/usr/local/bin` in order to invoke the command from anywhere.

	# sudo cp sendMail /usr/local/bin

# License

Unless otherwise noted, this code is licensed under the Apache 2.0 license,
as found in the LICENSE file.
