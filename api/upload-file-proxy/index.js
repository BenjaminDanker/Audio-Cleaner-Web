module.exports = async function (context, req) {
    context.log('Upload file proxy function processed a request.');

    // This function can be used as a proxy for upload operations
    // Currently just returning a placeholder response
    context.res = {
        status: 200,
        body: {
            message: "Upload file proxy is available"
        }
    };
};
