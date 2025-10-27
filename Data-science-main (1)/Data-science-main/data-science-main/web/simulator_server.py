#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
미사일 시뮬레이터 웹 서버
웹 인터페이스에서 시뮬레이터를 실행할 수 있도록 API 제공
"""

import os
import json
import subprocess
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import webbrowser

class SimulatorHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.getcwd(), **kwargs)

    def do_POST(self):
        if self.path == '/run-simulator':
            self.handle_run_simulator()
        else:
            self.send_error(404, "Not Found")

    def handle_run_simulator(self):
        try:
            # POST 데이터 파싱
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            script_name = data.get('script', '')

            if script_name not in ['main.py', 'main_6dof.py']:
                self.send_json_response(False, "잘못된 스크립트 이름입니다.")
                return

            # 스크립트 파일 존재 확인
            script_path = os.path.join(os.getcwd(), '..', 'Data-science', script_name)
            if not os.path.exists(script_path):
                self.send_json_response(False, f"스크립트 파일을 찾을 수 없습니다: {script_name}")
                return

            # 비동기로 시뮬레이터 실행
            def run_simulation():
                try:
                    print(f"시뮬레이터 실행 시작: {script_name}")

                    # Python 스크립트 실행
                    result = subprocess.run(
                        ['python', script_path],
                        cwd=os.path.dirname(script_path),
                        capture_output=True,
                        text=True,
                        timeout=300  # 5분 타임아웃
                    )

                    if result.returncode == 0:
                        print(f"시뮬레이터 실행 완료: {script_name}")
                        self.send_json_response(True, "시뮬레이터 실행 완료", {"output": result.stdout})
                    else:
                        print(f"시뮬레이터 실행 실패: {script_name}")
                        self.send_json_response(False, f"시뮬레이터 실행 실패: {result.stderr}")

                except subprocess.TimeoutExpired:
                    self.send_json_response(False, "시뮬레이터 실행 시간이 초과되었습니다.")
                except Exception as e:
                    self.send_json_response(False, f"시뮬레이터 실행 중 오류 발생: {str(e)}")

            # 비동기로 실행
            thread = threading.Thread(target=run_simulation)
            thread.daemon = True
            thread.start()

            # 즉시 응답 (실행 시작 확인)
            self.send_json_response(True, f"{script_name} 시뮬레이터 실행을 시작했습니다. 완료까지 잠시 기다려주세요.")

        except Exception as e:
            self.send_json_response(False, f"요청 처리 중 오류 발생: {str(e)}")

    def send_json_response(self, success, message, data=None):
        response = {
            'success': success,
            'message': message
        }
        if data:
            response.update(data)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        self.wfile.write(json.dumps(response, ensure_ascii=False, indent=2).encode('utf-8'))

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def run_server(port=8000):
    """웹 서버 실행"""
    print(f"미사일 시뮬레이터 웹 서버를 시작합니다... (포트: {port})")
    print("웹 브라우저에서 http://localhost:{} 로 접속하세요".format(port))

    # 자동으로 브라우저 열기
    def open_browser():
        time.sleep(1)  # 서버 시작 대기
        webbrowser.open(f'http://localhost:{port}')

    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    try:
        server = HTTPServer(('localhost', port), SimulatorHandler)
        print("서버가 실행 중입니다. Ctrl+C로 종료하세요.")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다...")
        server.shutdown()

if __name__ == "__main__":
    run_server(8000)
