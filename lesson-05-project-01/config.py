DEBUG = True  # 启动Flask的Debug模式
BCRYPT_LEVEL = 13  # 配置Flask-Bcrypt拓展
MAIL_FROM_EMAIL = "robert@example.com"  # 设置邮件来源
# Flask使用这个密钥来对cookies和别的东西进行签名。你应该在instance文件夹中设定这个值，并不要把它放入版本控制中。
SECRET_KEY = "klsajdghgkljasglkhasdfhasdgasdfg"
