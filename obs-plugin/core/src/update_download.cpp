// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/update_download.hpp"

#include "acw/update_verify.hpp"
#include "http_get.hpp"

#include <ixwebsocket/IXNetSystem.h>

#include <filesystem>
#include <fstream>

namespace acw {

namespace fs = std::filesystem;

struct UpdateDownload::Impl {
	detail::HttpGet http;
	std::atomic<uint64_t> done{0};
	std::atomic<bool> cancelled{false};
};

UpdateDownload::UpdateDownload() : impl_(std::make_shared<Impl>()) {}
UpdateDownload::~UpdateDownload() = default;

void UpdateDownload::cancel()
{
	impl_->cancelled = true;
	impl_->http.cancel();
}

uint64_t UpdateDownload::bytes_done() const
{
	return impl_->done.load();
}

DownloadResult UpdateDownload::run(const UpdateFile &file, const std::string &dest_dir, const DownloadOptions &opt)
{
	DownloadResult r;
	if (!valid_update_file_name(file.name) || file.size == 0 || file.size > kMaxUpdateFileBytes || !is_hex_digest(file.sha512, 128) ||
	    opt.base_url.empty() || opt.base_url.back() != '/') {
		r.error = "the update list names a file that cannot be used";
		return r;
	}
	const fs::path dir = fs::u8path(dest_dir);
	const fs::path dest = dir / fs::u8path(file.name);
	const fs::path part = dir / fs::u8path(file.name + ".part");
	std::error_code ec;
	fs::remove(dest, ec);
	fs::remove(part, ec);

	std::ofstream out;
	std::unique_ptr<Sha512> hash;
	uint64_t got = 0;
	bool write_failed = false, too_big = false;

	detail::HttpReceiver rx;
	rx.begin = [&] {
		// every request starts over: the body of a redirect or an error page must never end up in the file
		if (out.is_open())
			out.close();
		out.clear();
		out.open(part, std::ios::binary | std::ios::trunc);
		hash = std::make_unique<Sha512>();
		got = 0;
		impl_->done = 0;
		write_failed = !out.is_open();
		too_big = false;
	};
	rx.chunk = [&](const std::string &chunk) {
		if (write_failed)
			return false;
		got += chunk.size();
		if (got > file.size) {  // more than the list says: stop, the rest is never fetched
			too_big = true;
			return false;
		}
		out.write(chunk.data(), static_cast<std::streamsize>(chunk.size()));
		if (!out) {
			write_failed = true;
			return false;
		}
		hash->update(chunk.data(), chunk.size());
		impl_->done = got;
		return true;
	};

	detail::HttpGetOptions o;
	o.user_agent = opt.user_agent;
	o.accept = "application/octet-stream";
	o.ca_file = opt.ca_file;
	o.connect_timeout_s = opt.connect_timeout_s;
	o.transfer_timeout_s = opt.transfer_timeout_s;
	o.total_timeout_s = opt.transfer_timeout_s + opt.connect_timeout_s * (opt.max_redirects + 1);
	o.max_redirects = opt.max_redirects;
	o.allow_local_http = opt.allow_local_http;

	ix::initNetSystem();
	std::string error;
	const bool ok = impl_->http.get(opt.base_url + file.name, o, rx, error);
	ix::uninitNetSystem();
	if (out.is_open())
		out.close();

	auto fail = [&](const std::string &why) {
		fs::remove(part, ec);
		r.error = why;
		r.cancelled = impl_->cancelled.load();
		return r;
	};
	if (impl_->cancelled)
		return fail("cancelled");
	if (write_failed)
		return fail("the update could not be saved on this computer");
	if (too_big)
		return fail("the file on the server is not the one the update list describes");
	if (!ok) {
		const bool http_error = error.compare(0, 5, "HTTP ") == 0;
		return fail(http_error ? "the server does not have the update (" + error + ")" : "the download failed (" + error + ")");
	}
	if (got != file.size)
		return fail("the download was incomplete");
	if (hash->finish_hex() != file.sha512)
		return fail("the file on the server is not the one the update list describes");
	fs::rename(part, dest, ec);
	if (ec)
		return fail("the update could not be saved on this computer");
	r.ok = true;
	r.path = dest.u8string();
	return r;
}

}  // namespace acw
