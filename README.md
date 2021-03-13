# sendMail
A simple command-line tool that uses cURL since I always mess up the mailx configurations.
Note that this tools works only for gmail at the moment.

# How to use

As a first step you need to configure you account settings by using the following command:

	$ ./sendMail -c -a "youEmailaccount@gmail.com" -p "youEmailaccountsPassword"

If the above step is successful, then you can start sending emails:

	$ ./sendMail -x -t "receiversMail" -s "emailSubject" -m "emailBody"

You can also add the executable `sendMail` file under the `/usr/local/bin` in order to invoke the command from anywhere.

	# sudo mv sendMail /usr/local/bin
