from flask import Flask, render_template

app = Flask(__name__)


# @app.route('/')
# def hello_world():
#
#     return render_template("/test5/demo.html")

    # return render_template("/views/yanking.html")

# @app.route('/QIAN')
# def hello_world2():
#
#     return render_template("/test4/index.html")

    # return render_template("/views/yanking.html")


@app.route('/')
def hello_world2():

    return render_template("views/end.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0',port = '5000')
